import { ModeProvider } from "./service/ModeProvider";
import Header from "./Components/Header/Header";
import { useEffect, useState } from "react";
import Main from "./Components/Main/Main";
import "./App.css";

const App = () => {
  const [mode, setMode] = useState<string>("dark");
  const [isConnected, setIsConnected] = useState<boolean>(false);

  useEffect(() => {
    handleAppMode();
  }, [mode]);

  const handleAppMode = () => {
    let mode = localStorage.getItem("mode");
    if (mode) setMode(mode);
  };

  return (
    <ModeProvider value={{ mode, setMode }}>
      <div id="App" data-theme={mode}>
        <Header isConnected={isConnected} />
        <Main setIsConnected={setIsConnected} />
      </div>
    </ModeProvider>
  );
};
export default App;
