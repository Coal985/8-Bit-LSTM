# 8-Bit Music Generation: An LSTM Trained on NES Soundtracks

A PyTorch project that trains a stacked LSTM on notecharts from NES game soundtracks (NES-MDB) to generate new 8-bit-style music. A Gradio interface lets you generate clips and play them back, and output is rendered to WAV audio.

## Skills Demonstrated

The main purpose of this project was the process, not the quality of the generated music. It was a way to practice building a complete machine learning pipeline end to end:

Sequence modeling · multi-output neural network design · embedding layers for categorical data · sliding-window data preparation · model checkpointing · building an interactive interface (Gradio) · converting model output into audio

## Motivation

I built this as a personal skills project to get hands-on experience with sequential data, where the order of the data carries meaning and each step depends on what came before it. Music is a natural example of this, and NES chiptune is a compact, well-structured version of it. The project was a chance to work through the full pipeline for sequential data: framing a long stream of notes into overlapping windows, using an LSTM to learn temporal patterns across those windows, and turning the model's output back into audio you can actually listen to.

## Data & Methods

- **Source:** The [NES Music Database (NES-MDB)](https://github.com/chrisdonahue/nesmdb), using the "separated score" format, which is sampled at a fixed 24 Hz.
- **Representation:** Each timestep contains four voices of the NES sound chip: two pulse waves (`p1`, `p2`), triangle (`tr`), and noise (`no`). Each voice's values were remapped to a compact vocabulary and passed through its own embedding layer, then concatenated along the feature axis.
- **Sequence framing:** A sliding window of 32 timesteps with a stride of 8.
- **Model:** A PyTorch `nn.Module` with two stacked LSTM layers (LayerNorm and Dropout after each) and four separate output heads, one linear layer per voice, each sized to that voice's own vocabulary.
- **Training:** Adam optimizer (learning rate 0.0001), cross-entropy loss summed across the four voice heads, batch size 64, 10 epochs. The best checkpoint is saved as `best_model.pt`.
- **Audio rendering:** Generated scores are converted to VGM and rendered to WAV using [VGMPlay](https://github.com/vgmrips/vgmplay), built with MSYS2.

| Setting             | Value                                 |
| ------------------- | ------------------------------------- |
| Dataset             | NES-MDB, separated score (24 Hz)      |
| Window / stride     | 32 / 8                                |
| Architecture        | 2 stacked LSTMs + 4 per-voice heads   |
| Optimizer           | Adam, lr = 0.0001                     |
| Loss                | Cross-entropy, summed across 4 voices |
| Batch size / epochs | 64 / 10                               |

## Results

An example output is [full_generated.wav](https://github.com/Coal985/8-Bit-LSTM/blob/main/full_generated.wav).

## Limitations

- **Output quality:** The generated music is rough. It captures some of the texture of NES audio but lacks the long-range structure (repeating melodies, sections, and endings) of real game soundtracks.
- **Hardware limitations:** The small model, short context, and limited training were all driven by the hardware available for this project. Longer windows, larger models, and longer training runs require more memory and compute than I had access to.
- **Short context:** A 32-step window at 24 Hz covers only about 1.3 seconds of music, so the model has no way to learn phrase-level or song-level structure. Longer windows would help, but they increase memory use and training time.
- **Independent voice heads:** Each voice is predicted by its own output layer, so coordination between voices (harmony, rhythm locking) is only learned indirectly through the shared LSTM state.
- **Limited training:** A short training run on a single dataset, with no extensive hyperparameter tuning or architecture comparison, again constrained by available hardware.
- **Style control:** The model generates in the blended style of the whole dataset. Game- or genre-specific generation is not implemented.

## Future Work

Longer context windows or a transformer architecture, conditioning on game or style, joint modeling of voices, and more training and tuning.

## Tools & Stack

Python · PyTorch · NumPy · Gradio · Jupyter · NES-MDB · VGMPlay · MSYS2

## Repository Contents

- `8Bit_Music_Project.ipynb`: data preparation, model, training, and generation
- `8-Bit_interface.ipynb`: Gradio interface for generating and playing back music
- `best_model.pt`: trained model weights
- `full_generated.wav`: example generated audio

## Running It

1. Download the NES-MDB dataset (separated score format) from the link above and place it next to the notebooks.
2. Install dependencies: `pip install torch numpy gradio`
3. Run `8Bit_Music_Project.ipynb` to train, or load `best_model.pt` to skip training.
4. Run `8-Bit_interface.ipynb` to launch the Gradio app.
5. To render audio, build VGMPlay from source (MSYS2 provides the compiler on Windows) and follow its instructions for converting VGM files to WAV.

## Acknowledgments

- [NES-MDB](https://github.com/chrisdonahue/nesmdb) by Chris Donahue et al. for the dataset
- [VGMPlay](https://github.com/vgmrips/vgmplay) for audio rendering

## AI Disclosure

AI tools were used to assist with coding, debugging, and conceptual guidance during the development of this project, as well as in documenting it. All modeling decisions and final implementation were reviewed and validated by me.
