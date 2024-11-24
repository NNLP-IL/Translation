# Translation

This repo contains 3 services for a complete **translation** app: 
1. Inference  - [Readme](/translation-inference/Readme.md)
2. Client Side - [Readme](/translation-client/README.md)
3. Server Side - [Readme](/translation-be/README.md)

Each directory contains `README.md` file with instructions for setup and usage after clone the repository:

   ```bash
   git clone git@github.com:NNLP-IL/Translation.git
   cd yourrepository
   ```

## Usage Options
This repository offers two primary ways to interact with the application:

1. **Direct Model Inference, Training, and Evaluation:** The inference module allows you to:

   Run the model directly for inference tasks.

   Fine-tune the model with custom data.

   Evaluate the model's performance based on specific metrics.

2. **Web Application Interface:** Launch the full web application by following the provided instructions to set up both the server and client sides. Ensure to configure the necessary environment variables—specifically, the server's port and the client-side connection settings. This option provides a user-friendly interface for interacting with the application.


## About the App

#### Key Features
Bi-Directional Language Support:

Hebrew ↔ Arabic translation.
NER and transliteration for both languages.

This holistic translation system provides comprehensive language services, including:

* Translation: Supports Hebrew ↔ Arabic translations.
* Named Entity Recognition (NER): Identifies entities in text for both languages.
* Transliteration: Converts text between scripts, ensuring phonetic consistency across languages.

All models used for translation and NER are configurable and can be replaced with Hugging Face models via environment variables.

## License

The app is released under the [Apache License](https://www.apache.org/licenses/LICENSE-2.0).