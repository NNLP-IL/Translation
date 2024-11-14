import { enterTextPlaceholders, error404Placeholders, error422Placeholders, LanguageCode, translateTextPlaceholders } from "./BoxesTranlationPH";

const textFieldStyle = {
  "& .MuiInputBase-input": {
    maxHeight: "100%",
    paddingLeft: "16px",
    height: "100%!important",
    overflowY: "auto!important",
  },
  "& .MuiInputBase-root": {
    lineHeight: 1.5,
    height: "35.5vh",
    borderRadius: "8px",
    padding: "20px 20px 40px 60px;",
    color: "var(--elementsText)!important",
  },
  "& .Mui-disabled": {
    backgroundColor: "initial!important",
  },
  "& .MuiInputBase-inputMultiline": {
    color: "var(--elementsText)",
    "&::placeholder": {
      color: "var(--elementsText)!important",
      fontSize: "1.6em",
    },
    "-webkit-text-fill-color": "var(--elementsText)!important",
  },
  "& .MuiOutlinedInput-root": {
    "& fieldset": {
      borderColor: "var(--textFieldBorder)", // Change border color
    },
    "&:hover fieldset": {
      borderColor: "var(--elementsText)", // Change border color on hover
    },
    "&.Mui-focused fieldset": {
      borderColor: "var(--elementsText)", // Change border color when focused
    },
    "& input": {
      color: "var(--elementsText)", // Change text color
    },
  },
};

const handleTextFieldPh = (code: string): string => {
  if (Object.values(LanguageCode).includes(code as LanguageCode)) {
    return enterTextPlaceholders[code as LanguageCode];
  }
  return ""; // Default placeholder for unknown languages
};


const handleTranslateFieldPh = (code: string) => {
  if (Object.values(LanguageCode).includes(code as LanguageCode)) {
    return translateTextPlaceholders[code as LanguageCode];
  }
  return ""; // Default placeholder for unknown languages
};

const handleErrorMsgText = (code: string, errorStatus: number) => {
  if (errorStatus === 422) {
    if (Object.values(LanguageCode).includes(code as LanguageCode)) {
      return error422Placeholders[code as LanguageCode];
    }
    return ""; // Default placeholder for unknown languages
  } else if (errorStatus === 404) {
    if (Object.values(LanguageCode).includes(code as LanguageCode)) {
      return error404Placeholders[code as LanguageCode];
    }
    return ""; // Default placeholder for unknown languages
  }
};


const textLengthStyle = {
  mt: 1,
  textAlign: "center",
  fontFamily: "Monospace",
  position: "absolute",
  bottom: "6px",
  left: "24px",
  fontSize: "1.1em",
  color: "var(--textFieldBorder)",
};

const sourceMaxLength = 5000;

const iconStyle = {
  fontSize: "1.9em",
  marginTop: "-16px",
  color: "var(--iconsColor)",
};

const clearIconStyle = {
  top: "30px",
  left: "22px",
  position: "absolute",
  color: "var(--iconsColor)",
  "&.Mui-disabled": {
    color: "var(--textFieldBorder)!important",
  },
};
const checkboxStyle = {
  "& .Mui-disabled": {
    color: "#636b74!important",
  },
  color: "var(--elementsText)",
  "& .MuiTypography-root": { fontSize: "0.9em" },
  "& .MuiCheckbox-root": {
    color: "#1976d2",
  },
};
export {
  textFieldStyle,
  handleTranslateFieldPh,
  handleTextFieldPh,
  textLengthStyle,
  sourceMaxLength,
  iconStyle,
  clearIconStyle,
  checkboxStyle,
  handleErrorMsgText,
};
