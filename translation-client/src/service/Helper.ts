import { LanguageInterface, languageInterface } from "../Types";
import languageList from "./limitedLanguageList.json";
import { diffWords } from "diff";

const getCurrentLanguage = (key: string, value: string) => {
  let currentLanguage = languageList.filter((item: any) => item[key] === value);
  return currentLanguage[0];
};
const handleCompareAndUpdate = (oldText: string, newText: string) => {
  const diff = diffWords(oldText, newText);
  let result = "";
  diff.forEach((part: any) => {
    if (part.added) {
      result += part.value;
    } else if (!part.removed) {
      result += part.value;
    }
  });
  return result;
};

const getLanguageByCode = (code: string) => {
  // if (typeof(code) === 'undefined') return;
  const fieldName: keyof languageInterface = "code";
  const result: LanguageInterface =
    languageList.find((item) => item[fieldName] === code) || languageList[0];
  return result;
};

export { getCurrentLanguage, handleCompareAndUpdate, getLanguageByCode };
