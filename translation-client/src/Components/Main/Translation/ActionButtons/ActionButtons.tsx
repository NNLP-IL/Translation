import { buttonStyle, buttonsData } from "./ActionButtonsConfig";
import ListItem from "@mui/joy/ListItem";
import Checkbox from "@mui/joy/Checkbox";
import { Tooltip } from "@mui/material";
import List from "@mui/joy/List";
import "./ActionButtons.css";

const ActionButtons = (props: {
  selectedBtn: string;
  disabledBtns: boolean;
  setSelectedBtn: (arg0: string) => void;
}) => {
  const { selectedBtn, setSelectedBtn, disabledBtns } = props;

  return (
    <div role="group" aria-labelledby="topping">
      <List
        sx={{
          "--List-gap": "16px",
          "--ListItem-radius": "10px",
        }}
        orientation="horizontal"
      >
        {buttonsData.map((item, i) => (
          <ListItem key={i} sx={{ width: "120px", justifyContent: "center" }}>
            <Tooltip title={item.title ? item.title : ""} placement="top">
              <Checkbox
                overlay
                disableIcon
                variant="solid"
                sx={buttonStyle}
                label={item.title}
                disabled={disabledBtns}
                checked={selectedBtn === item.text}
                onChange={() => setSelectedBtn(item.text)}
              />
            </Tooltip>
          </ListItem>
        ))}
      </List>
    </div>
  );
};

export default ActionButtons;
