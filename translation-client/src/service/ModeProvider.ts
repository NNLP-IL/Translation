import { createContext } from "react";

export type GlobalContent = {
  mode: string;
  setMode: (arg0: string) => void;
};

const ModeContext = createContext<GlobalContent>({
  mode: "",
  setMode: () => {},
});

export const ModeProvider = ModeContext.Provider;
export default ModeContext;
