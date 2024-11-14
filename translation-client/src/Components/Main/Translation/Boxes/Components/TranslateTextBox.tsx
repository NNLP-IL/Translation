import React from "react";
import { Grid, TextField } from "@mui/material";
import { handleTranslateFieldPh } from "../BoxesConfig";
import DotLoader from "../../../../../Commons/Loader/DotLoader";
import CustomCopyIcon from "../../../../../Commons/CopyIcon/CopyIcon";
import { TranslateTextBoxProps } from "../../../../../Types";

const TranslateTextBox = ({
  textFieldStyle,
  source,
  handleTranslateViewText,
  showLoader,
  isStreaming,
  translate,
}: TranslateTextBoxProps) => {
  return (
    <Grid
      item
      xs={12}
      md={5.8}
      order={{ xs: 3, md: 3 }}
      sx={{ position: "relative" }}
    >
      <TextField
        disabled
        multiline
        fullWidth
        id="translate"
        variant="filled"
        value=""
        sx={{
          ...textFieldStyle,
          borderRadius: "8px",
          backgroundColor: "var(--appBackground)",
        }}
        placeholder={handleTranslateFieldPh(source.code)}
        InputProps={{
          readOnly: true,
          inputComponent: () => (
            <div
              style={{
                margin: 0,
                height: "100%",
                width: "100%",
                maxHeight: "100%",
                overflowY: "auto",
                paddingTop: "4px",
                paddingRight: "16px",
                whiteSpace: "pre-wrap",
                color: "var(--elementsText)!important",
              }}
            >
              <p
                style={{
                  margin: 0,
                  direction: source.direction === "rtl" ? "rtl" : "ltr",
                }}
              >
                {handleTranslateViewText()}
                {showLoader && <DotLoader showAnimation={!isStreaming} />}
              </p>
            </div>
          ),
        }}
      />
      <CustomCopyIcon value={translate} languageCode={source.code} />
    </Grid>
  );
};

export default TranslateTextBox;
