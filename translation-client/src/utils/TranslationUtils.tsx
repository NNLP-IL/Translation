import { ColorMapResponse, Entity } from "../Types";

import { Tooltip } from "@mui/material";

const GetColorMapDesc = (tag: string, colorMap: ColorMapResponse) => {
  let tagResultDesc: string = "";
  if (tag) {
    if (Object.keys(colorMap).indexOf(tag.replace("B-", "")) !== -1) {
      tagResultDesc = colorMap[tag.replace("B-", "")].description;
    } else if (Object.keys(colorMap).indexOf(tag.replace("I-", "")) !== -1) {
      tagResultDesc = colorMap[tag.replace("I-", "")].description;
    }
  }
  return tagResultDesc;
};

const flattenEntitiesByWordAndOffset = (entities: Entity[]) => {
  const uniqueEntitiesMap = new Map();

  entities.forEach(entity => {
    if (Array.isArray(entity.offset)) {
      entity.offset.forEach(([start, end]) => {
        const key = `${entity.word}-${start}-${end}`;
        if (!uniqueEntitiesMap.has(key)) {
          uniqueEntitiesMap.set(key, {
            word: entity.word,
            tag: entity.tag,
            tag_hex: entity.tag_hex,
            offset: [start, end],
            source: entity.source
          });
        }
      });
    }
  });

  return Array.from(uniqueEntitiesMap.values()).sort((a, b) => a.offset[0] - b.offset[0]);
}


/**
 * Generates a styled JSX sentence with spans based on entity offsets.
 * @param sentence The input sentence string.
 * @param entities Array of entities containing word, tag, tag_hex, and offset.
 * @param colorMap Array of data about the mapping of entites.
 * @returns JSX.Element[] with styled and plain text spans.
 */
const GetStyledResultBasedOnEntities = (
  sentence: string,
  entities: Entity[],
  colorMap: ColorMapResponse
): any => {
  entities = entities.filter((obj) => obj.offset.length > 0); // Filter all entites without offset - dosn't appear on the sentence
  const uniqueOffsets = flattenEntitiesByWordAndOffset(entities);
  

  const sentenceComponents: JSX.Element[] = [];
  let currentIndex = 0;

  uniqueOffsets.forEach(({ offset, tag_hex, word, tag }, index) => {
    let [start, end] = offset;

    // Add plain text before the styled span
    if (currentIndex < start) {
      sentenceComponents.push(
        <span key={`plaintext-${currentIndex}-${end}`}>
          {sentence.slice(currentIndex, start)}
        </span>
      );
    }

    const tagDesc = GetColorMapDesc(tag, colorMap);

    const newStyledWord = (
      <Tooltip
        key={`tooltip-${index}-${start}-${end}`}
        title={`${tagDesc}, Word: ${word}`}
      >
        <span
          style={{
            padding: "4px 10px",
            borderRadius: "30px",
            border: `2px solid ${tag_hex}`,
            cursor: "pointer",
          }}
        >
          {sentence.slice(start, end)}
        </span>
      </Tooltip>
    );

    // Add the styled span for the entity
    sentenceComponents.push(newStyledWord);

    currentIndex = end;
  });

  // Add remaining plain text after the last entity
  if (currentIndex < sentence.length) {
    sentenceComponents.push(
      <span key={`plaintext-${currentIndex}`}>
        {sentence.slice(currentIndex)}
      </span>
    );
  }

  return sentenceComponents;
};

export { GetStyledResultBasedOnEntities };
