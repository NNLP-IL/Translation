import React from "react";
import { Chip, Grid, IconButton, TextField, Tooltip, Typography } from "@mui/material";
import { handleTextFieldPh } from "../BoxesConfig";
import CloseIcon from "@mui/icons-material/Close";
import { DocumentScannerOutlined } from "@mui/icons-material";
import { SourceTextBoxProps } from "../../../../../Types";

const SourceTextBox = ({
  updateTextboxText,
  handleClearIcon,
  textFieldStyle,
  source,
  sourceMaxLength,
  textLengthStyle,
  text,
  clearIconStyle,
  showLoader,
  nerCheckbox,
  selectedBtn,
  setNerCheckbox,
  sourceTextBoxEnabled,
}: SourceTextBoxProps) => {
  return (
    <>
      <Grid
        item
        xs={12}
        md={5.8}
        sx={{ position: "relative" }}
        order={{ xs: 3, md: 3 }}
      >
        <TextField
          id="text"
          disabled={!sourceTextBoxEnabled}
          multiline
          fullWidth
          value={text}
          onChange={(e) => {
            updateTextboxText("base", e.target.value);
            if (e.target.value === "") return handleClearIcon();
          }}
          sx={{
            ...textFieldStyle,
            direction: source.direction,
          }}
          inputProps={{ sourceMaxLength }}
          placeholder={handleTextFieldPh(source.code)}
        />
        <Typography variant="h6" gutterBottom sx={textLengthStyle}>
          {text?.length} / {sourceMaxLength}
        </Typography>
        <Tooltip title="Clear" sx={clearIconStyle}>
          <IconButton disabled={showLoader} onClick={handleClearIcon}>
            <CloseIcon sx={{ fontSize: "1.2em" }} />
          </IconButton>
        </Tooltip>
      </Grid>
      <Grid item xs={12} md={12} order={{ xs: 3, md: 3 }}>
        <Tooltip title="Named Entity Recognition" leaveDelay={200} arrow>
          <Chip
            icon={<DocumentScannerOutlined fontSize="small" />}
            color="info"
            label="NER"
            variant={nerCheckbox ? "filled" : "outlined"}
            clickable={true}
            disabled={selectedBtn === "Transliterate"}
            onClick={() => setNerCheckbox(!nerCheckbox)}
            sx={{ padding: "10px" }}
          />
        </Tooltip>
      </Grid>
    </>
  );
};

export default SourceTextBox;
