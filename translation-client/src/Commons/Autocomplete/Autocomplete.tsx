import { textFieldStyle } from "./AutocompleteConfig";
import Autocomplete from "@mui/material/Autocomplete";
import TextField from "@mui/material/TextField";
import { languageInterface } from "../../Types";

const CustomAutoComplete = (props: {
  onChange: any;
  disabled: boolean;
  value: languageInterface;
  options: languageInterface[];
}) => {
  return (
    <Autocomplete
      size={"small"}
      disablePortal
      value={props.value}
      id="combo-box-demo"
      options={props.options}
      readOnly={props.disabled}
      onChange={props.onChange}
      sx={{ ...textFieldStyle, width: "10%" }}
      renderInput={(params) => <TextField {...params} label="Language" />}
    />
  );
};

export default CustomAutoComplete;
