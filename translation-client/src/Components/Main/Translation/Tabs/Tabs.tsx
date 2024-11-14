import { topGridStyle } from "./TabsConfig";
import languageList from "../../../../service/limitedLanguageList.json";
import { Chip, Grid } from "@mui/material";
import { CustomTabsPropsTypes } from "../../../../Types";
import ActionButtons from "../ActionButtons/ActionButtons";

const CustomTabs = (props: CustomTabsPropsTypes) => {
  const {
    source,
    target,
    updateLanguages,
    showLoader,
    handleSwap,
    selectedBtn,
    setSelectedBtn,
  } = props;

  return (
    <Grid
      item
      container
      direction="row"
      sx={topGridStyle}
      alignItems="center"
      order={{ xs: 2, md: 1 }}
      flexDirection="row-reverse"
      justifyContent="space-between"
    >
      <Grid item>
        <Grid container direction="row" justifyContent="flex-start" spacing={2}>
          {languageList.map((item, index) => {
            return (
              <Grid item>
                <Chip
                  color="primary"
                  sx={{ padding: "10px" }}
                  clickable={true}
                  key={index}
                  label={item.label}
                  variant={source.label === item.label ? "filled" : "outlined"}
                  onClick={() => {
                    if (target.code === item.code) {
                      handleSwap();
                    } else {
                      updateLanguages("source", item);
                    }
                  }}
                />
              </Grid>
            );
          })}
        </Grid>
      </Grid>

      <Grid item>
        <ActionButtons
          disabledBtns={showLoader}
          selectedBtn={selectedBtn}
          setSelectedBtn={setSelectedBtn}
        />
      </Grid>

      <Grid item>
        <Grid container direction="row" justifyContent="flex-start" spacing={2}>
          {languageList.map((item, index) => {
            return (
              <Grid item>
                <Chip
                  color="primary"
                  sx={{ padding: "10px" }}
                  clickable={true}
                  key={index}
                  label={item.label}
                  variant={target.label === item.label ? "filled" : "outlined"}
                  onClick={() => {
                    if (source.code === item.code) {
                      handleSwap();
                    } else {
                      updateLanguages("source", item);
                    }
                  }}
                />
              </Grid>
            );
          })}
        </Grid>
      </Grid>
    </Grid>
  );
};

export default CustomTabs;
