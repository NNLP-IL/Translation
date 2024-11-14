import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import ModeIcon from "../../../Commons/ModeIcon/ModeIcon";
import TranslateIcon from "@mui/icons-material/Translate";
import Popup from "./Popup";
import ServerStatus from "../../../Commons/ServerStatus/ServerStatus";
import { CustomAppBarProps } from "../../../Types";

export default function CustomAppBar({ isConnected }: CustomAppBarProps) {
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar
        position="static"
        sx={{
          color: "var(--elementsText)",
          bgcolor: "var(--elementsBackground)",
        }}
      >
        <Toolbar>
          <TranslateIcon sx={{ fontSize: "1.9em" }} />
          <Typography
            variant="h5"
            noWrap
            component="div"
            sx={{
              display: { xs: "none", sm: "block" },
              margin: "0px 4px",
              letterSpacing: "0.6px",
            }}
          >
            המתורגמן
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Box
            sx={{ display: { xs: "none", md: "flex", alignItems: "center" } }}
          >
            <ModeIcon />
            <ServerStatus language={"he"} isConnected={isConnected} />
            {/* <Popup text={"v0.1.0"} /> */}
          </Box>
        </Toolbar>
      </AppBar>
    </Box>
  );
}
