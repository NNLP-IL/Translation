const buttonStyle = {
  // "& .MuiCheckbox-action": {
  //   backgroundColor: "#ffffff",
  //   color: "#00000"
  // },
  // "& .JoyCheckbox-root": {
  //   color: "#00000"
  // },
  "& .Mui-checked": {
    backgroundColor: "#1976d2c4",
    boxShadow: "rgba(3, 102, 214, 0.3) 0px 0px 0px 3px",
  },
  "& .MuiCheckbox-label": {
    fontSize: "0.9em",
  },
  "& .Mui-disabled": {
    opacity: 0.3,
  },
};

const buttonsData = [
  { text: "Translate", title: "תרגום" },
  { text: "Transliterate", title: "תעתיק" },
];
export { buttonStyle, buttonsData };
