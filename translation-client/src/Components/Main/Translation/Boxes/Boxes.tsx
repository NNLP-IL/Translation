import { useEffect, useState } from "react";
import {
  Grid,
} from "@mui/material";
import service from "../../../../service/request";
import {
  ColorMapResponse,
  customBoxesProps,
  Entity,
} from "../../../../Types";
import {
  sourceMaxLength,
  textFieldStyle,
  textLengthStyle,
  clearIconStyle,
  handleErrorMsgText,
} from "./BoxesConfig";
import useDebounce from "../../../../utils/useDebounce";
import { searchTermDelay } from "../TranslationConfig";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { handleCompareAndUpdate } from "../../../../service/Helper";
import { GetStyledResultBasedOnEntities } from "../../../../utils/TranslationUtils";
import SourceTextBox from "./Components/SourceTextBox";
import TranslateTextBox from "./Components/TranslateTextBox";

import "./Boxes.css";

const CustomBoxes = (props: customBoxesProps) => {
  const {
    text,
    data,
    source,
    translate,
    showLoader,
    selectedBtn,
    setShowLoader,
    updateTextboxText,
    streamedText,
    setStreamedText,
    setIsConnected,
  } = props;

  const debouncedSearchTerm = useDebounce(text, searchTermDelay);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [nerCheckbox, setNerCheckbox] = useState<boolean>(true);
  const [errorMsg, setErrorMsg] = useState<JSX.Element | string>("");
  const [colorMap, setColorMap] = useState<ColorMapResponse>({});
  const [sourceTextBoxEnabled, setSourceTextBoxEnabled] = useState<boolean>(true);

  useEffect(() => {
    handleColorMap();
  }, []);

  useEffect(() => {
    if (selectedBtn === "Transliterate") {
      setNerCheckbox(false);
    }
    if (!text) return;
    makeRequest();
  }, [debouncedSearchTerm, selectedBtn, source]);

  useEffect(() => {
    updateTextboxText("result", "");
    setStreamedText([]);
  }, [selectedBtn]);

  const simulateStreaming = (
    currentSentence: string,
    entities: Entity[],
    useDelay: boolean
  ) => {
    entities.length === 0 ? setNerCheckbox(false) : setNerCheckbox(true);
    if (!currentSentence) return;
    setIsStreaming(true);
    const createdEntities = GetStyledResultBasedOnEntities(
      currentSentence,
      entities,
      colorMap
    );

    setStreamedText((prev) => [...prev, ...createdEntities, <> </>]);
    setIsStreaming(false);
    setShowLoader(false);
  };

  const makeRequest = async () => {
    setShowLoader(true);
    setErrorMsg("");
    setSourceTextBoxEnabled(false);
    switch (selectedBtn) {
      case "Translate":
        await handleTranslateBtn();
        setSourceTextBoxEnabled(true);
        break;
      case "Transliterate":
        await handleTransliterateBtn();
        setSourceTextBoxEnabled(true);
        break;
      default:
        break;
    }
  };

  const handleTranslateBtn = async () => {
    const ctrl = new AbortController();
    const SERVER = process.env.REACT_APP_API_URL;
    let sentence = "";
    let entities: any = [];
    let offset = 0;

    await fetchEventSource(`${SERVER}/api/translator/translate_sentences`, {
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      method: "POST",
      signal: ctrl.signal,
      openWhenHidden: true,
      body: JSON.stringify({
        sentences: text,
        entity_alignment: true,
        translate_direction: `${data.source}2${data.target}`,
      }),
      async onopen(res) {
        handleErrors(res?.status);
      },
      async onmessage(ev) {
        let data = JSON.parse(ev.data);
        if (!data) return setShowLoader(false);

        sentence += data?.sentence + " \n";
        data.entities.forEach((entity: any) => {
          if (Array.isArray(entity.offset) && entity.offset.length > 0) {
            // If offset contains nested arrays (like [[42, 45]])
            if (Array.isArray(entity.offset[0])) {
              entity.offset = entity.offset.map((range: [number, number]) => [
                range[0] + offset,
                range[1] + offset,
              ]);
            }
            // If offset is just an array of numbers (like [30, 35])
            else if (typeof entity.offset[0] === "number") {
              entity.offset = [
                entity.offset[0] + offset,
                entity.offset[1] + offset,
              ];
            }
          }
        });

        entities = [...entities, ...data?.entities];
        offset += sentence.length - offset;

        handleTranslateInputChange(sentence, entities);
      },
      onclose() {
        setShowLoader(false);
      },
      onerror(err) {
        setShowLoader(false);
      },
    });
  };

  const handleTransliterateBtn = async () => {
    try {
      let response = await service.getTransliteration(data);
      if (!response?.data) return setShowLoader(false);
      simulateStreaming(response?.data, [], true);
      updateTextboxText("result", response?.data);
    } catch (error: any) {
      handleErrors(error?.response?.status);
    }
  };

  const handleErrors = (errorStatus: any) => {
    if (!errorStatus) {
      return setShowLoader(false);
    } else if (errorStatus === 404 || errorStatus === 422) {
      setErrorMsg(
        <span>
          {handleErrorMsgText(source.code, errorStatus)}
          <span style={{ color: "#cb666a", marginRight: 4 }}> ●</span>
        </span>
      );
      setShowLoader(false);
    }
  };

  const handleTranslateInputChange = async (sentence: any, entities: any) => {
    let newText: string = handleCompareAndUpdate(translate, sentence);
    updateTextboxText("result", "");
    setStreamedText([]);
    simulateStreaming(newText, entities, false);
    updateTextboxText("result", newText);
  };

  const handleColorMap = async () => {
    const colorMap: ColorMapResponse = await service.getColorMap();
    if (colorMap) setIsConnected(true);
    setColorMap(colorMap);
  };

  const handleClearIcon = () => {
    updateTextboxText("base", "");
    setErrorMsg("");
    updateTextboxText("result", "");
    setStreamedText([]);
  };

  const handleTranslateViewText = () => {
    if (errorMsg) return errorMsg;
    if (selectedBtn === "Translate") {
      if (nerCheckbox) {
        return streamedText;
      } else {
        return translate;
      }
    } else if (selectedBtn === "Transliterate") {
      return streamedText;
    }
  };

  return (
    <>
      <TranslateTextBox
      textFieldStyle={textFieldStyle}
      source={source}
      handleTranslateViewText={handleTranslateViewText}
      showLoader={showLoader}
      isStreaming={isStreaming}
      translate={translate}
      />
      <Grid item xs={12} md={0.4} order={{ xs: 3, md: 3 }}>
      </Grid>
      <SourceTextBox
        updateTextboxText={updateTextboxText}
        handleClearIcon={handleClearIcon}
        textFieldStyle={textFieldStyle}
        source={source}
        sourceMaxLength={sourceMaxLength}
        textLengthStyle={textLengthStyle}
        text={text}
        clearIconStyle={clearIconStyle}
        showLoader={showLoader}
        nerCheckbox={nerCheckbox}
        selectedBtn={selectedBtn}
        setNerCheckbox={setNerCheckbox}
        sourceTextBoxEnabled={sourceTextBoxEnabled}
      />
    </>
  );
};

export default CustomBoxes;
