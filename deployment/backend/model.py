from pytorch_lightning import LightningModule
from torch import nn, cat, squeeze, sigmoid, optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LitGRU(LightningModule):
    def __init__(
        self,
        vocab,
        dimension=64,
        sequenceDepth=64,
        dropoutWithinLayers=0.5,
        dropoutOutput=0.5,
        learningRate=1e-3,
        amyloid=True,
        lossFunc=nn.SmoothL1Loss(),
    ):
        super(LitGRU, self).__init__()
        self.embedding = nn.Embedding(len(vocab), sequenceDepth)
        self.dimension = dimension
        self.gru = nn.GRU(
            input_size=sequenceDepth,
            hidden_size=dimension,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropoutWithinLayers,
        )
        self.dropOnOutput = nn.Dropout(p=dropoutOutput)
        self.learningRate = learningRate
        self.fc = nn.Linear(2 * dimension, 1)
        self.loss_func = lossFunc
        self.amyloid = amyloid
        self.save_hyperparameters(ignore=["lossFunc", "vocab"])

    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(
            text_emb, text_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, : self.dimension]
        out_reverse = output[:, 0, self.dimension :]
        out_reduced = cat((out_forward, out_reverse), 1)
        text_fea = self.dropOnOutput(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = squeeze(text_fea, 1)
        text_out = sigmoid(text_fea)
        return text_out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learningRate)
        return optimizer

    def training_step(self, batch, batch_idx):
        header, sequence, sequence_length, prionLabel, amyloidLabel = batch
        output = self(sequence, sequence_length)
        loss = self.loss_func(
            output, prionLabel if not self.hparams["amyloid"] else amyloidLabel
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch,
    ):
        header, sequence, sequence_length, prionLabel, amyloidLabel = batch
        output = self(sequence, sequence_length)
        val_loss = self.loss_func(
            output, prionLabel if not self.hparams["amyloid"] else amyloidLabel
        )
        self.log("val_loss", val_loss, prog_bar=True)
        return {"val_loss": val_loss}

    def test_step(
        self,
        batch,
    ):
        header, sequence, sequence_length, prionLabel, amyloidLabel = batch
        output = self(sequence, sequence_length)
        test_loss = self.loss_func(
            output, prionLabel if not self.hparams["amyloid"] else amyloidLabel
        )
        self.log("test_loss", test_loss, prog_bar=True)
        return {"test_loss": test_loss}

    def predict_step(self, batch):
        sequence, sequence_length = batch
        sequence = sequence.to(self.device)
        sequence_length = sequence_length.to(self.device)
        output = self(sequence, sequence_length)
        return output
