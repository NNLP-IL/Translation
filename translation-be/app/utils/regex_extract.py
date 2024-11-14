import re
import sys
import os
from typing import List, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from objects.entities import Entity, EntityTagSource

class RegexUtil:
    @staticmethod
    def extract_emails(text: str, tag: str = "EMAIL", tag_hex: Optional[str] = None):
        """ Regular expression for extracting emails in English, Hebrew, and Arabic """
        emails: List[Entity] = []
        email_pattern = re.compile(
            r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[\u0590-\u05FF0-9_.+-]+@[\u0590-\u05FF0-9-]+\.[\u0590-\u05FF0-9-.]+|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9_.+-]+@[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9-]+\.[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF0-9-.]+'
        )
        for email in email_pattern.finditer(text):
            emails.append(Entity(word=email.group(), tag=tag, tag_hex=tag_hex, offset=email.span(), source=EntityTagSource.REGEX.value))
        return emails
    
    @staticmethod
    def extract_id_numbers(text: str, tag: str = "ID", tag_hex: Optional[str] = None):
        """
        Extracts ID numbers (sequence of digits of length 9 to 12)
        This pattern matches both Western and Arabic digits
        """
        ids: List[Entity] = []
        id_pattern = re.compile(
            r'\b(\d{9,12}|[\u0660-\u0669]{9,12})\b'
        ) 
        for _id in id_pattern.finditer(text):
            ids.append(Entity(word=_id.group(), tag=tag, tag_hex=tag_hex, offset=_id.span(), source=EntityTagSource.REGEX.value))
        return ids
        
    @staticmethod
    def extract_phone_numbers(text: str, tag: str = "PHONE", tag_hex: Optional[str] = None):
        """
        Extracts phone numbers
        This pattern matches various phone number formats with Western and Arabic digits
        """
        phones: List[Entity] = []
        phone_pattern = re.compile(
            r'\+?[\d\u0660-\u0669][\d\u0660-\u0669\s\-()]{7,15}[\d\u0660-\u0669]'
        ) 
        for phone in phone_pattern.finditer(text):
            phones.append(Entity(word=phone.group(), tag=tag, tag_hex=tag_hex, offset=phone.span(), source=EntityTagSource.REGEX.value))
        return phones
    
    @classmethod
    def extract_entities(cls, text: str, tag_map: dict = {}):
        return cls.extract_emails(text=text, tag_hex=tag_map.get("EMAIL",{}).get("hex")) + cls.extract_id_numbers(
            text=text, tag_hex=tag_map.get("ID",{}).get("hex")) + cls.extract_phone_numbers(text=text, tag_hex=tag_map.get("EMAIL",{}).get("hex"))
        
# Example usage:
if __name__ == "__main__":
    sample_text = """
    This is a sample text with an email: example@example.com.
    Here is an ID number: 123456789.
    البريد الإلكتروني: مثال@مثال.كوم
    כתובת דוא"ל: דוגמא@דוגמא.קום
    رقم الهاتف: +971 50 123 4567
    מספר טלפון: 03-1234567
    Phone number: (123) 456-7890
    Another ID number: 9876543210.
    رقم تعريف آخر: ١٢٣٤٥٦٧٨٩٠١١
    هاتف آخر: +٩٧١ ٥٠ ١٢٣ ٤٥٦٧
    """
    print(RegexUtil.extract_entities(text=sample_text))