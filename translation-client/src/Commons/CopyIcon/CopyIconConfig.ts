import { LanguageCode } from "../../Components/Main/Translation/Boxes/BoxesTranlationPH";
import { successCopyResultPlaceholders, failedCopyResultPlaceholders } from "./CopyIconTranslationPH";

const succesCopyResultFieldPh = (code: string) => {
    if (Object.values(LanguageCode).includes(code as LanguageCode)) {
        return successCopyResultPlaceholders[code as LanguageCode];
    }
    return ""; // Default placeholder for unknown languages
};

const failedCopyResultFieldPh = (code: string) => {
    if (Object.values(LanguageCode).includes(code as LanguageCode)) {
        return failedCopyResultPlaceholders[code as LanguageCode];
    }
    return ""; // Default placeholder for unknown languages
};

export {
    succesCopyResultFieldPh,
    failedCopyResultFieldPh,
}