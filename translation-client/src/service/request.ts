import axios from "axios";
import { ColorMapResponse } from "../Types";
const SERVER = process.env.REACT_APP_API_URL;

const languageDetection = async (words: string) => {
  return await axios.get(`${SERVER}/api/detection/${words}`);
};

const translatorSentences = async (data: {
  sentences: string;
  translate_direction: string;
}) => {
  try {
    return await axios.post(`${SERVER}/api/translator/translate_sentences`, {
      ...data,
      entity_alignment: true,
    });
  } catch (err: any) {
    console.log("Error:", err);
    throw err;
  }
};

const getTransliteration = async (data: {
  text: string;
  source: string;
  target: string;
}) => {
  const { text, source, target } = data;
  try {
    return await axios.post(
      `${SERVER}/api/transliterate?source=${source}&target=${target}`,
      {
        content: text,
      }
    );
  } catch (err) {
    console.log("Error:", err);
    throw err;
  }
};

const getColorMap: () => Promise<ColorMapResponse> = async () => {
  try {
    const res = await axios.get(`${SERVER}/api/entity_extract/tagging_map`);
    return res?.data;
  } catch (err) {
    console.log("Error:", err);
    // throw err;
  }
};

const service = {
  languageDetection,
  translatorSentences,
  getTransliteration,
  getColorMap,
};
export default service;
