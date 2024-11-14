import { LanguageCode } from "../../Components/Main/Translation/Boxes/BoxesTranlationPH";

const successCopyResultPlaceholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "הועתק!",
    [LanguageCode.English]: "Copied!",
    [LanguageCode.Arabic]: "تم النسخ",
};

const failedCopyResultPlaceholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "נכשל!",
    [LanguageCode.English]: "Failed!",
    [LanguageCode.Arabic]: "فاشل",
};

export {
    successCopyResultPlaceholders,
    failedCopyResultPlaceholders,
}