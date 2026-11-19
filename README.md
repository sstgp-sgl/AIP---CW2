# AIP304 Coursework 2 · Option 2

Does binary lesion-mask supervision improve rejection of unseen diseases on familiar plant hosts while maintaining known-class macro-F1?

## Run the notebook

Use `CW2.ipynb` for the coursework. Install dependencies with `python -m pip install -r requirements.txt`, then run the notebook from the repository root. It downloads [PlantSeg v7](https://zenodo.org/records/17719108), or you can place the release archive at `data/raw/plantseg.zip`. Set `DOWNLOAD=False` to use existing data and cached metadata offline.

The current notebook verifies the dataset, audits the nine-host subset (4,349 images, 43 classes), and generates draft A/B manifests. See [audit results](artifacts/audit/) and [split protocol](artifacts/splits/split_protocol.json). The next milestone is to resolve cross-partition duplicates, review training-mask exclusions, and freeze both manifests before training.

## Data and split rules

Use PlantSeg v7 and keep its original partitions. The nine hosts are apple, banana, citrus, corn, potato, soybean, tomato, wheat, and zucchini. Each host–disease pair is one class.

| Class role | Original training images | Original validation images | Original test images |
| --- | --- | --- | --- |
| Known | Used for training | Used for validation | Used for testing |
| Development unknown | Unused | Used for validation | Unused |
| Test unknown | Unused | Unused | Used for testing |

- **Both assignments** use citrus as development unknown and potato as test unknown. For each remaining host, sort disease labels alphabetically. **A** holds out the first label for development and the second for testing. **B** holds out the second for development and the third for testing. All remaining labels are known.
- Record image IDs, labels, class roles, and exclusions. Keep exact duplicates within one original partition. Document any unresolved near-duplicate risk. Use the same eligible samples for both methods within each assignment.
- Use binary masks only for supervision on known training images. Validation and test inference use RGB only. Fit data-dependent preprocessing on training images. Use deterministic evaluation transforms and record software, hardware, seeds, initial weights, and runtime.

## Option 2 requirements

- **Research design.** Review at least three closely related methods. Before training, state a falsifiable hypothesis and justify the allowable drop in known-class macro-F1.
- **Models and runs.** Compare one focused lesion-supervised method with a matched classifier. Match backbone, initialisation source, preprocessing, augmentation, and training and search budgets. Compare maximum softmax, maximum logit, and energy scores. Run the required ablations and use three seeds per assignment for the main comparison. Train each assignment independently.
- **Validation.** Select checkpoints by known validation macro-F1. Select configurations by familiar-host validation AUOSCR within the predefined macro-F1 tolerance. Choose the primary score on baseline validation data and apply the same rule to both models. Set the operating threshold to accept about 95% of known validation images. Freeze all choices for both assignments before inspecting either test result.
- **Test reporting.** Report known macro-F1 and accuracy before rejection, unknown AUROC, AUOSCR, and the three fixed-threshold rates (known acceptance, correct-and-accepted known rate, and unknown false-acceptance rate). Use familiar-host AUOSCR as the primary outcome. Report the unfamiliar-host control separately with the same known images. Break down shared versus unseen disease labels where sample sizes permit. Report sample sizes, seed means, standard deviations, paired differences, and representative failures.

## Submission

Due **13 December at 23:59** (Week 13). Submit one package per team of 1–3 students.

- **Notebook.** One reproducible `.ipynb` that runs in a fresh Colab or Kaggle environment, includes full training code, and regenerates the report figures and tables.
- **Report.** One PDF of at most five pages, including references, AI-use disclosure, and team contributions where applicable. State the chosen option and list names and student IDs.
- **Video.** At most three minutes. Show working inference, one result, and one limitation. Show the presenter’s face during narration and include every team member.

Full requirements and grading rubric are in the [coursework task sheet](AIP304%20CW2%20Task%20Sheet.pdf), especially Sections 2 and 4–7.
