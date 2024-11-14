import { getLanguageByCode } from "../../../service/Helper";

const boxesContainerStyle = {
  padding: "30px",
  maxWidth: "90vw",
  maxHeight: "90vh",
  paddingTop: "10px",
  alignItems: "center",
  position: "relative",
  boxSizing: "initial",
  borderRadius: "10px",
  marginTop: "initial",
  backgroundColor: "var(--elementsBackground)",
  "& .MuiFormControl-root": { height: "100%" },
};

const iconStyle = {
  fontSize: "1.9em",
  color: "var(--elementsText)",
};

const initialLanguage = {
  label: "Arabic",
  code: "ar",
};

const initialSource: string = "he";

const initialTarget: string = "ar";

// const initialSource = {
//   label: "Arabic",
//   code: "ar",
//   direction: "rtl",
//   native_name: "العربية",
// };

// const initialTarget = {
//   label: "Hebrew",
//   code: "he",
//   direction: "rtl",
//   native_name: "Hebrew",
// };

const initialTabList = [getLanguageByCode(initialSource).label, getLanguageByCode(initialTarget).label];

const searchTermDelay: number = 1500;

export {
  iconStyle,
  boxesContainerStyle,
  initialLanguage,
  initialSource,
  initialTabList,
  initialTarget,
  searchTermDelay,
};
