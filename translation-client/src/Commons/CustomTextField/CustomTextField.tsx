import { TextField } from "@mui/material";
import { textFieldStyle } from "./CustomTextFieldConfig";

const CustomTextField = (
  id: string,
  style: any,
  variant: any,
  value: string,
  onChange: any,
  placeholder: string
) => {
  return (
    <TextField
      disabled
      multiline
      fullWidth
      id={id}
      variant={variant}
      value={value}
      sx={{
        ...textFieldStyle,
        style,
      }}
      onChange={onChange}
      placeholder={placeholder}
    />
  );
};

export default CustomTextField;
