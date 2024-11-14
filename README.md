# Translation

This repo contains 3 services for a complete **translation** app: 
1. Inference  - [Readme](/translation-inference/README.md)
2. Client Side - [Readme](/translation-client/README.md)
3. Server Side - [Readme](/translation-be/README.md)

Each directory contains `README.md` file with instructions for setup and usage after clone the repository:

   ```bash
   git clone git@github.com:NNLP-IL/Translation.git
   cd yourrepository
   ```

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