const textFieldStyle = {
  "& .MuiInputBase-root": {
    height: "35.5vh",
  },
  "& .MuiInputBase-inputMultiline": {
    height: "100%!important",
    color: "var(--elementsText)!important",
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

export { textFieldStyle };
