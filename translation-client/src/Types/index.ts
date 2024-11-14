export interface languageInterface {
  label: string;
  code: string;
}

export interface CustomTabsPropsTypes {
  source: LanguageInterface;
  target: LanguageInterface;
  updateLanguages: any;
  showLoader: boolean;
  handleSwap: () => void;
  selectedBtn: string;
  setSelectedBtn: (arg0: string) => void;
}

export interface LanguageInterface {
  label: string;
  code: string;
  direction: string;
  native_name: string;
}

export interface customBoxesProps {
  text: string;
  value_s: number;
  translate: string;
  showLoader: boolean;
  selectedBtn: string;
  updateTextboxText: any;
  source: LanguageInterface;
  streamedText: JSX.Element[];
  setStreamedText: React.Dispatch<React.SetStateAction<JSX.Element[]>>;
  setText?: (arg0: any) => void;
  handleDetectLanguageMode: () => void;
  setShowLoader: (arg0: boolean) => void;
  data: { text: string; source: string; target: string };
  setIsConnected: any;
}

export interface DrowerInterface {
  type: string;
  open: boolean;
}

export interface Entity {
  word: string;
  tag: string;
  tag_hex: string;
  offset: [number, number][];
  source: string;
}

export interface ColorMapObj {
  description: string;
  color: string;
  hex: string;
}

export interface ColorMapResponse {
  [key: string]: ColorMapObj;
}

export interface SourceTextBoxProps {
  updateTextboxText: any;
  handleClearIcon: () => void;
  textFieldStyle: any;
  source: LanguageInterface;
  sourceMaxLength: number;
  textLengthStyle: any;
  text: string;
  clearIconStyle: any;
  showLoader: boolean;
  nerCheckbox:boolean;
  selectedBtn: string;
  setNerCheckbox: (arg0: boolean) => void;
  sourceTextBoxEnabled: boolean;
}

export interface TranslateTextBoxProps {
  textFieldStyle: any,
  source: LanguageInterface,
  handleTranslateViewText: any,
  showLoader: boolean,
  isStreaming: boolean,
  translate: string
}


export interface MainProps {
  setIsConnected: (arg0: boolean) => void;
}

export interface TranslationProps {
  setIsConnected: (arg0: boolean) => void;
}

export interface HeaderProps {
  isConnected: boolean;
}


export interface CustomAppBarProps {
  isConnected: boolean;
}