import { IconButton, Tooltip } from "@mui/material";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";
import { useState } from "react";
import { failedCopyResultFieldPh, succesCopyResultFieldPh } from "./CopyIconConfig";

interface CustomCopyIconProps {
  value: string;
  languageCode: string;
}

const CustomCopyIcon = ({ value, languageCode }: CustomCopyIconProps) => {
  const [status, setStatus] = useState<"idle" | "success" | "error">("idle");

  const handleClick = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setStatus("success");
    } catch (err) {
      setStatus("error");
    }

    // Reset status after 5 seconds
    setTimeout(() => setStatus("idle"), 1000);
  };

  const getIcon = () => {
    switch (status) {
      case "success":
        return <CheckCircleIcon sx={{ color: "green" }} />;
      case "error":
        return <ErrorIcon sx={{ color: "red" }} />;
      default:
        return <ContentCopyIcon />;
    }
  };

  return (
    <Tooltip
      title={
        status === "success"
          ? succesCopyResultFieldPh(languageCode)
          : status === "error"
          ? failedCopyResultFieldPh(languageCode)
          : "Copy"
      }
      sx={{
        left: "20px",
        bottom: "20px",
        position: "absolute",
        transition: "border 0.3s ease-in-out",
        border:
          status === "success"
            ? "2px solid green"
            : status === "error"
            ? "2px solid red"
            : "none",
        borderRadius: "50%",
      }}
    >
      <IconButton onClick={handleClick}>{getIcon()}</IconButton>
    </Tooltip>
  );
};

export default CustomCopyIcon;
