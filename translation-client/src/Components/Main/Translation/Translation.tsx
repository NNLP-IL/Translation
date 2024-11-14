import { Grid } from "@mui/material";
import CustomTabs from "./Tabs/Tabs";
import CustomBoxes from "./Boxes/Boxes";
import { useEffect, useState } from "react";
import service from "../../../service/request";
import { getCurrentLanguage, getLanguageByCode } from "../../../service/Helper";
import {
  LanguageInterface,
  TranslationProps,
} from "../../../Types";
import {
  initialSource,
  initialTarget,
  boxesContainerStyle,
} from "./TranslationConfig";

const Translation = ({ setIsConnected }: TranslationProps) => {
  const [text, setText] = useState<string>("");
  const [lngValues, setLngValues] = useState<{
    value_s: number;
    value_t: number;
  }>({
    value_s: 0,
    value_t: 1,
  });
  const [textboxTexts, setTextboxText] = useState<{
    base: string;
    result: string;
  }>({
    base: "",
    result: "",
  });
  const [showLoader, setShowLoader] = useState<boolean>(false);
  const [selectedBtn, setSelectedBtn] = useState<string>("Translate");
  const [languages, setLanguages] = useState<{
    target: LanguageInterface;
    source: LanguageInterface;
  }>({
    target: getLanguageByCode(initialTarget),
    source: getLanguageByCode(initialSource),
  });
  const [sourceTabList, setSourceTabList] = useState<string[]>([]);
  const [streamedText, setStreamedText] = useState<JSX.Element[]>([]);

  useEffect(() => {
    handleInitialSourceTabList();
  }, []);

  useEffect(() => {
    setSourceTabList([languages.target.label, languages.source.label]);
  }, [languages]);

  const swapLngValues = () => {
    setLngValues((prevValues) => ({
      value_s: prevValues.value_t,
      value_t: prevValues.value_s,
    }));
  };

  const swapLanguages = () => {
    setLanguages((prevValues) => ({
      source: prevValues.target,
      target: prevValues.source,
    }));
  };

  const swapTextBoxText = () => {
    setTextboxText((prevValues) => prevValues);
  };

  // Generalized function to update any property in the state object of lngValues
  const updateLngValue = <Key extends keyof typeof lngValues>(
    key: Key,
    newValue: (typeof lngValues)[Key]
  ) => {
    setLngValues((prevValues) => ({
      ...prevValues,
      [key]: newValue,
    }));
  };

  // Generalized function to update any property in the state object of languages
  const updateLanguages = <Key extends keyof typeof languages>(
    key: Key,
    newValue: (typeof languages)[Key]
  ) => {
    setLanguages((prevValues) => ({
      ...prevValues,
      [key]: newValue,
    }));
  };

  // Generalized function to update any property in the state object og languages
  const updateTextboxText = <Key extends keyof typeof textboxTexts>(
    key: Key,
    newValue: (typeof textboxTexts)[Key]
  ) => {
    setTextboxText((prevValues) => ({
      ...prevValues,
      [key]:
        key === "result" && newValue !== ""
          ? prevValues.result + newValue
          : newValue,
    }));
  };

  const handleInitialSourceTabList = () => {
    let list = localStorage.getItem("initialSourceTabList");
    if (!list) return;
    let initialList = list.split(",");
    setSourceTabList(initialList);
  };

  const handleDetectLanguageMode = async () => {
    if (!text) return;
    try {
      let { data } = await service.languageDetection(text);
      let currentLanguage = getCurrentLanguage("code", data.data.language);

      updateLanguages("source", currentLanguage);
    } catch (err) {
      console.log("Error:", err);
    }
  };

  const handleSwap = () => {
    swapLanguages();
    swapLngValues();
    swapTextBoxText();
    setStreamedText([]);
  };

  return (
    <Grid container rowSpacing={2} spacing={1} sx={boxesContainerStyle}>
      <CustomTabs
        source={languages.source}
        target={languages.target}
        updateLanguages={updateLanguages}
        handleSwap={handleSwap}
        showLoader={showLoader}
        selectedBtn={selectedBtn}
        setSelectedBtn={setSelectedBtn}
      />
      <CustomBoxes
        text={textboxTexts.base}
        source={languages.source}
        value_s={lngValues.value_s}
        updateTextboxText={updateTextboxText}
        streamedText={streamedText}
        setStreamedText={setStreamedText}
        translate={textboxTexts.result}
        showLoader={showLoader}
        selectedBtn={selectedBtn}
        setShowLoader={setShowLoader}
        handleDetectLanguageMode={handleDetectLanguageMode}
        data={{
          text: textboxTexts.base,
          source: languages.source.code,
          target: languages.target.code,
        }}
        setIsConnected={setIsConnected}
      />
    </Grid>
  );
};

export default Translation;
