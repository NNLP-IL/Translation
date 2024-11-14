enum LanguageCode {
    Hebrew = "he",
    English = "en",
    Arabic = "ar",
}

const translateTextPlaceholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "תרגום",
    [LanguageCode.English]: "Translation",
    [LanguageCode.Arabic]: "بدلا من ذلك",
};

const enterTextPlaceholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "יש להזין טקסט",
    [LanguageCode.English]: "Please enter text",
    [LanguageCode.Arabic]: "الرجاء إدخال النص",
};

const error422Placeholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "הטקסט אינו תואם לשפת המקור",
    [LanguageCode.English]: "The text isn't in the source language",
    [LanguageCode.Arabic]: "النص لا يتطابق مع اللغة الأصلية",
};

const error404Placeholders: Record<LanguageCode, string> = {
    [LanguageCode.Hebrew]: "תעתוק לא נמצא, מודל נכשל",
    [LanguageCode.English]: "Transliterate failed",
    [LanguageCode.Arabic]: "لم يتم العثور على النسخة، فشل النموذج",
};


export {
    translateTextPlaceholders,
    enterTextPlaceholders,
    error422Placeholders,
    error404Placeholders,
    LanguageCode,
}