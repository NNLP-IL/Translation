"use client";
import { Fade } from "@mui/material";
import { HashLoader } from "react-spinners";
import "./Loader.css";

const CustomLoader = (props: {
  width?: string;
  showBackDrop?: boolean;
  localBackdrop?: boolean;
}) => {
  const { localBackdrop, showBackDrop, width } = props;
  if (showBackDrop || localBackdrop) {
    return (
      <Fade in={true} timeout={1000}>
        <div className={showBackDrop ? "backdrop" : "localBackdrop"}>
          <div className="loaderContainer">
            <HashLoader size={width || 50} color="var(--loaderColor)" />
          </div>
        </div>
      </Fade>
    );
  } else {
    return (
      <div className="loaderContainer">
        <HashLoader size={width || 50} color="var(--loaderColor)" />
      </div>
    );
  }
};

export default CustomLoader;
