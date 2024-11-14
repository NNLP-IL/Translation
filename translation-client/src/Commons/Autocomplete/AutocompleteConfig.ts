const textFieldStyle = {
  "& .MuiInputLabel-root": {
    color: "var(--elementsText)",
    "&.Mui-focused": {
      color: "var(--elementsText)!important",
    },
  },
  "& .MuiButtonBase-root": {
    color: "var(--elementsText)",
  },
  "& .MuiOutlinedInput-root": {
    "& fieldset": {
      borderColor: "var(--elementsText)", // Change border color
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

export { textFieldStyle };
