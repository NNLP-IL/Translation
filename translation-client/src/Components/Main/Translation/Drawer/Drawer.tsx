import "./Drawer.css";
import { listStyle } from "./DrawerConfig";
import { DrowerInterface, LanguageInterface } from "../../../../Types";
import languageList from "../../../../service/limitedLanguageList.json";
import {
  List,
  Fade,
  Grid,
  ListItem,
  Checkbox,
  ListItemButton,
  ListItemIcon,
  ListItemText,
} from "@mui/material";

const CustomDrawer = (props: {
  setCountry: any;
  source: LanguageInterface;
  handleSourceTabList: (arg0: string) => void;
  setOpenDrawer: (arg0: DrowerInterface) => void;
}) => {
  const { source, setCountry, setOpenDrawer, handleSourceTabList } = props;

  return (
    <Grid item xs={12} md={12} order={{ xs: 2, md: 2 }}>
      <Fade in={true}>
        <div className="customDrawer">
          <List sx={listStyle}>
            {languageList.map((language: LanguageInterface) => {
              const labelId = `checkbox-list-label-${language.label}`;
              return (
                <ListItem
                  disablePadding
                  key={language.code}
                  sx={{ width: "16.6%" }}
                >
                  <ListItemButton
                    dense
                    role={undefined}
                    onClick={() => {
                      setOpenDrawer({ open: false, type: "" });
                      setCountry(language);
                      handleSourceTabList(language.label);
                    }}
                  >
                    <ListItemIcon>
                      <Checkbox
                        edge="start"
                        checked={source.label.indexOf(language.label) !== -1}
                        tabIndex={-1}
                        disableRipple
                        inputProps={{ "aria-labelledby": labelId }}
                        sx={{ color: "var(--elementsText)" }}
                      />
                    </ListItemIcon>
                    <ListItemText id={labelId} primary={`${language.label}`} />
                  </ListItemButton>
                </ListItem>
              );
            })}
          </List>
        </div>
      </Fade>
    </Grid>
  );
};

export default CustomDrawer;
