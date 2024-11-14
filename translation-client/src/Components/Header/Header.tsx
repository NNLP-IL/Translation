import { HeaderProps } from "../../Types";
import CustomAppBar from "./AppBar/AppBar";
import "./Header.css";

const Header = ({ isConnected }: HeaderProps) => {
  return (
    <div id="Header">
      <div className="headerContainer">
        <CustomAppBar isConnected={isConnected} />
      </div>
    </div>
  );
};

export default Header;
