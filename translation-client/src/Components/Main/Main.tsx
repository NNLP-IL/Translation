import Translation from "./Translation/Translation";
import { MainProps } from "../../Types";

import "./Main.css";

const Main = ({setIsConnected}: MainProps) => {
  return (
    <div id="Main">
      <div className="mainContainer">
        <Translation setIsConnected={setIsConnected} />
      </div>
    </div>
  );
};

export default Main;
