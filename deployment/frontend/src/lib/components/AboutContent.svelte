<script lang="ts">
  const citation = `@misc{
  ejmockler2021aggrucheck,
  author = {Eric Jing Mockler},
  title = {Gated recurrent units classify amyloidogenic proteins from sequence alone},
  year = {2021},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.13147167},
}`;
</script>

<div class="about-content">
  <p class="intro">Identify and visualize amyloidogenic domains in proteins using gated recurrent units</p>
  
  <ul class="faq-list">
    <li>
      <i>What does it analyze?</i>
      <p>
        The tool analyzes protein sequences to identify regions with amyloidogenic potential. 
        It examines multiple window sizes and uses ensemble predictions to provide both 
        position-specific scores and confidence measures.
      </p>
    </li>
    <li>
      <i>How do I interpret the results?</i>
      <div class="faq-answer">
        <p>Results are displayed as an interactive visualization where:</p>
        <ul class="sub-list">
          <li>Colors indicate amyloid propensity (green → yellow → red)</li>
          <li>Hover over residues to see detailed predictions</li>
          <li>Confidence scores reflect agreement across model ensemble</li>
          <li>Window analysis shows local structural tendencies</li>
        </ul>
      </div>
    </li>
    <li>
      <i>How accurate is it?</i>
      <p>
        The model uses an ensemble of 3 GRU networks, each trained on different data splits. 
        Confidence scores indicate prediction reliability, with higher values (>0.5) suggesting 
        stronger agreement across the ensemble. The sliding window analysis helps identify 
        biologically relevant amyloidogenic domains.
      </p>
    </li>
    <li>
      <i>What inputs are accepted?</i>
      <div class="faq-answer">
        <p>You can input:</p>
        <ul class="sub-list">
          <li>Raw amino acid sequences</li>
          <li>FASTA format (with headers)</li>
          <li>Multiple sequences at once (up to 5)</li>
          <li>FASTQ format</li>
        </ul>
      </div>
    </li>
    <li>
      <i>Can I export the results?</i>
      <p>
        Yes! Use the "Download Results" button to get a CSV file containing position-specific 
        predictions, confidence scores, and sequence information for all analyzed proteins.
      </p>
    </li>
    <li>
      <i>Where can I learn more?</i>
      <p>
        Check out the <a href="https://github.com/ejmockler/AgGRU-Check" target="_blank" rel="noopener noreferrer">GitHub</a>
        and <a href="https://github.com/ejmockler/AgGRU-Check/blob/main/details.pdf" target="_blank" rel="noopener noreferrer">technical paper</a>
        for implementation details and to run the model locally.
      </p>
    </li>
    <li>
      <i>How do I cite this?</i>
      <pre class="citation">{citation}</pre>
    </li>
  </ul>
</div>

<style lang="scss">
  .about-content {
    .intro {
      font-size: 1.1rem;
      color: #264653;
      margin-bottom: 2rem;
      line-height: 1.4;
    }

    .faq-list {
      list-style: none;
      padding: 0;
      margin: 0;

      li {
        margin-bottom: 1.75rem;

        i {
          display: block;
          font-size: 1.1rem;
          font-weight: 600;
          color: #2A9D8F;
          margin-bottom: 0.75rem;
        }

        p {
          margin: 0;
          color: rgba(38, 70, 83, 0.9);
          line-height: 1.5;
        }

        .faq-answer {
          color: rgba(38, 70, 83, 0.9);
          line-height: 1.5;

          p {
            margin: 0 0 0.5rem 0;
          }
        }

        .sub-list {
          list-style: disc;
          margin: 0.5rem 0 0 1.5rem;
          
          li {
            margin-bottom: 0.25rem;
            font-size: 0.95rem;
          }
        }
      }
    }

    .citation {
      background: rgba(38, 70, 83, 0.1);
      padding: 1rem;
      border-radius: 6px;
      font-size: 0.85rem;
      overflow-x: auto;
      margin-top: 0.5rem;
    }

    a {
      color: #2A9D8F;
      text-decoration: none;
      font-weight: 500;
      
      &:hover {
        text-decoration: underline;
      }
    }
  }
</style> 