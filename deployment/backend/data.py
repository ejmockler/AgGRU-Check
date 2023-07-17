from torch.nn.utils.rnn import pad_sequence
from torch import tensor


from pydantic import BaseModel, validator
from typing import List


def is_fastq(seq: str) -> bool:
    lines = seq.strip().split("\n")
    if len(lines) % 4 != 0:
        return False
    for i in range(0, len(lines), 4):
        if not lines[i].startswith("@") or not lines[i + 2].startswith("+"):
            return False
        if len(lines[i + 1].split("\n")) != len(lines[i + 3].split("\n")):
            return False
    return True


def is_fasta(seq: str) -> bool:
    lines = seq.strip().split("\n")
    # FASTA sequences start with ">"
    if not lines[0].startswith(">"):
        return False
    # Sequence must not be empty
    if len(lines) <= 1:
        return False
    return True


class ProteinInput(BaseModel):
    sequences: List[str]

    @validator("sequences", each_item=True)
    def parse_sequence(cls, sequence):
        amino_acid_vocab = set("ACDEFGHIKLMNPQRSTVWY")
        lines = sequence.strip().split("\n")
        processed_sequence = ""

        if is_fastq(sequence):
            processed_sequence = "".join(
                line for i, line in enumerate(lines) if i % 4 == 1
            )
        elif is_fasta(sequence):
            processed_sequence = "".join(
                line for line in lines if not line.startswith(">")
            )
        else:
            processed_sequence = "".join(lines)

        if len(processed_sequence) > 34350:
            raise ValueError(
                f"A sequence of {len(processed_sequence)} characters is too long. The largest known protein chain is titin, which has up to ~34350 amino acids."
            )
        sequence_chars = set(processed_sequence.upper())
        invalid_chars = sequence_chars.difference(amino_acid_vocab)
        if invalid_chars:
            raise ValueError(
                f"The sequence contains invalid amino acids: {', '.join(invalid_chars)}. Canonical amino acids are: {','.join(amino_acid_vocab)}"
            )
        return processed_sequence


def collate_fn(batch):
    headers, sequences, prions, amyloids = zip(*batch)

    # Compute lengths of sequences
    sequence_lengths = tensor([len(seq) for seq in sequences])

    # Pad sequences and convert to tensor
    sequences_padded = pad_sequence(
        [tensor(seq) for seq in sequences], batch_first=True
    )

    # Convert targets to tensor
    prions = tensor(prions)
    amyloids = tensor(amyloids)

    return headers, sequences_padded, sequence_lengths, prions, amyloids
