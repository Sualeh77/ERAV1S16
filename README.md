# Transformer to translate EN to FR.

## Data Source and Preprocessing 
- **Dataset:** The dataset used for this task was the `OPUS book` translation dataset from hugging face.
- **Data Preprocessing:**
  - Implemented dynamic padding to handle variable-length sequences efficiently.
  - Removed English sentences that exceeded 150 characters.
  - Removed French sentences whose length was greater than the corresponding English sentence length plus 10 characters.

## Model Architecture 
- **Encoder-Decoder:** This model utilizes the encoder-decoder architecture for sequence-to-sequence translation tasks.
- **Parameter Sharing:** Implemented parameter sharing with a dense feedforward layer size (dff) of `1024.`
  - Sharing Pattern : 
    - [e1, e2, e3, e1, e2, e3] - for encoder
    - [d1, d2, d3, d1, d2, d3] - for decoder

## Training Configuration 
- **Batch Size:** Training was conducted with a batch size of `32.`
- **Epochs:** The training process spanned `20 epochs.`
- Used CrossEntrophy loss