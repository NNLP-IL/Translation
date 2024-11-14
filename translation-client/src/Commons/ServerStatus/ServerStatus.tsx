import React from "react";
import { Box, Typography, Tooltip } from "@mui/material";

type ServerStatusProps = {
  isConnected: boolean;
  language: "en" | "he" | "ar";
};

const translations = {
  en: { connected: "Connected", disconnected: "Not Connected" },
  he: { connected: "מחובר", disconnected: "לא מחובר" },
  ar: { connected: "Connecté", disconnected: "Non Connecté" },
};

const ServerStatus: React.FC<ServerStatusProps> = ({
  isConnected,
  language,
}) => {
  const statusText = isConnected
    ? translations[language]?.connected || "Connected"
    : translations[language]?.disconnected || "Not Connected";

  return (
    <Box display="flex" alignItems="center">
      <Tooltip title={statusText}>
        <Box
          sx={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            bgcolor: isConnected ? "green" : "red",
            marginRight: 5,
            marginLeft: 1,
          }}
        />
      </Tooltip>
      <Typography variant="body1">{statusText}</Typography>
    </Box>
  );
};

export default ServerStatus;
