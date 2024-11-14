from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_distance
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scripts.utils.print_colors import *


def compute_cosine_similarity(embedding1, embedding2):
    # Using Scipy to calculate cosine similarity
    similarity = 1 - cosine_distance(embedding1, embedding2)
    return similarity


class Chunker():
    def __init__(self, tokenizer=None, chunk_size=60, embedder_name='symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli'):
        # Load external tokenizer
        self.tokenizer = tokenizer
        # Create a RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer,
                                                                                       chunk_size=chunk_size,
                                                                                       chunk_overlap=0,
                                                                                       separators=['\n', '.', '،'],
                                                                                       keep_separator=False)
        self.chunk_size = chunk_size
        # Embedder
        self.embedder = SentenceTransformer(embedder_name)

    def combine_sentences(self, array_text, threshold):
        i = 0
        while i < len(array_text) - 1:
            try:
                embeddings_1 = self.embedder.encode(array_text[i])
                embeddings_2 = self.embedder.encode(array_text[i + 1])
                if compute_cosine_similarity(embeddings_1, embeddings_2) >= threshold:
                    # Combine sentences by merging the next sentence into the current one
                    array_text[i] = array_text[i] + " " + array_text[i + 1]
                    # Remove the next sentence since it's now combined into the current one
                    del array_text[i + 1]
                    # Do not increment i to check the newly combined sentence with its next neighbor
                    continue
            except IndexError:
                # This catch is for safety, but in normal operations, we should not hit an Index Error due to the loop's condition
                print("Reached the end of the array_text. Exiting.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                break
            # Only increment i if sentences were not combined, to move on to the next pair
            i += 1
        return array_text

    def split_sentences_by_space(self, sentences):
        """
        Splits sentences into multiple sentences based on the limit of tokens.
        Parameters:
        - sentences (List[str]): The original sentences to be split.
        Returns:
        List[str]: A list containing the original or split sentences.
        """
        total_chunks = []
        # Iterate over each sentence in the input list
        for sen in sentences:
            # Tokenize the input sentence
            tokens = self.tokenizer.tokenize(sen)
            # If the number of tokens is within the limit, keep the sentence as is
            if len(tokens) <= self.chunk_size:
                total_chunks.append(sen)
            else:
                # If the number of tokens exceeds the limit, split the sentence by words into chunks
                current_chunk = []
                current_size = 0
                words = sen.split()  # Splitting sentence into words
                for word in words:
                    word_tokens = self.tokenizer.tokenize(word)  # Tokenize the current word to check its size
                    if current_size + len(word_tokens) <= self.chunk_size:
                        # If adding this word doesn't exceed the limit, add it to the current chunk
                        current_chunk.append(word)
                        current_size += len(word_tokens)
                    else:
                        # If the current word exceeds the limit, save the current chunk and start a new one
                        total_chunks.append(' '.join(current_chunk))  # Join the words to form a sentence
                        current_chunk = [word]  # Start a new chunk with the current word
                        current_size = len(word_tokens)  # Reset the size counter to the current word's size
                # Don't forget to add the last chunk if it's not empty
                if current_chunk:
                    total_chunks.append(' '.join(current_chunk))
        return total_chunks

    def split_text2chunks(self, text: str):
        # Split the text to paragraphs by '\n'
        paragraphs = text.split('\n')
        paragraphs = [item for item in paragraphs if item]

        # Split each paragraph into chunks
        total_chunks, paragraphs_counter = [], []
        for idx, paragraph in enumerate(paragraphs):
            senetnces = paragraph.split(".")
            # Remove empty and add '.'
            senetnces = [f"{item}." for item in senetnces if item]
            print(len(senetnces))
            # Combine using the embeddings
            cmb_splitted_src = self.combine_sentences(senetnces, 0.35)
            # split if chunk is larger than the token limit
            paragraph_chunks = []
            for chunk in cmb_splitted_src:
                new_chunks = self.text_splitter.split_text(chunk)
                # in case there are no punctuations split by space
                new_chunks = self.split_sentences_by_space(new_chunks)
                paragraph_chunks.extend(new_chunks)
            total_chunks.extend(paragraph_chunks)
            paragraphs_counter.extend([idx] * len(paragraph_chunks))

        # Display chunks
        if self.tokenizer:
            for i, chunk in enumerate(total_chunks):
                print(
                    f"{PRINT_START}{GREEN}Chunk {i + 1} [{len(self.tokenizer.encode(chunk, add_special_tokens=False))} tokens]:"
                    f"\n{chunk}\n {PRINT_STOP}")

        return total_chunks, paragraphs_counter


def main():
    src = '''
    عندما تُفتتح الرواية الأولى من السلسلة، هاري بوتر وحجر الفيلسوف، من الواضح أن أمرًا مهمًّا قد حَدَثَ في عالم السحرة - وهو أمرٌ رائع جدًا لدرجة أنَّ الماغل (غير السحرة) لاحظوا علامات عليه. يتم الكشف عن الخلفية الكاملة لهذا الحدث وماضي هاري بوتر تدريجياً خلال السلسلة. بعد الفصل التمهيدي، يقفز الكتاب إلى وقتٍ قصيرٍ قبل عيد ميلاد هاري بوتر الحادي عشر، وفي هذه المرحلة يبدأ الكشف عن حقيقة كَوْنِهِ ساحرًا.
    
    على الرغم من محاولات فيرنون درسلي - زوج خالة هاري - منع الطفل من التعرف على قدراته السحرية إلَّا أنَ جهوده لا تُجدي نفعًا. حيث يلتقي هاري بنصف عملاق، اسمه روبياس هاغريد، ليحصل بذلك أول اتصال له بالعالم السحري. يُعَرِّفْ هاجريد عن نفسه بأنه أمين المفاتيح وحارس الأراضي في مدرسة هوجورتس للسحر والشعوذة، كما يحكي لهاري بعض الأشياء عن ماضيه. يَعْلَم هاري أنه عندما كان طفلاً، شَهِدَ مقتل والديه على يد ساحر الظلام المهووس بالسلطة اللورد فولدمورت (المعروف أكثر في المجتمع السحري باسم «أنت تعرف من» أو «الذي يجب أن لا نذكر إسمه»، ويُناديه ألبوس دمبلدور باسمه الحقيقي توم مارفولو ريدل) بعد قتل والديه ألقى فولدمورت تعويذة قاتلة على هاري لقتله. بدلاً من ذلك، حَدَثَ ما لم يكن متوقعًا حيث نجا هاري مع ندبةٍ على شكل برقٍ في جبهته كعلامةٍ على الهجوم، واختفى فولدمورت بعد ذلك بفترةٍ وجيزةٍ، ضعيفًا جدًّا بعد أن ارتدَّت عليه تعويذته التي أطلقها على هاري.
    
    باعتباره المُنقذ من عهد فولدمورت الرهيب، أصبح هاري أسطورةً حيةً في عالم السحرة. ومع ذلك، بناءً على أوامر الساحر الموقَّر والمعروف ألبوس دمبلدور، تم وضع هاري اليتيم في منزل أقاربه الماغل، عائلة درسلي، الذين أبقوه آمنًا لكنهم عاملوه معاملةً سيئةً، مثل سَجْنِهِ في خزانةٍ تحت الدرج وحرمانه من وجبات الطعام ومعاملته كخادم لهم. قام هاجريد بدعوة هاري رسميًا للالتحاق بمدرسة هوجورتس للسحر والشعوذة، وهي مدرسةٌ سحريةٌ شهيرةٌ في اسكتلندا تقوم بتعليم السحرة المراهقين أثناء مسيرتهم الدراسية لمدة سبع سنوات، من سن الحادية عشرة إلى السابعة عشرة.
    
    بمساعدة هاجريد، يستعِدُّ هاري للسنة الأولى من دراسته في هوجورتس حيث يشتري كتبه ولوازمه الدراسية بعد حصوله على ثروته الصغيرة التي تركها له والداه في بنك السحرة غرينغوت. عندما يبدأ هاري في استكشاف العالم السحري، يتعرف القارئ على العديد من المواقع الأساسية المُسْتَخْدَمَة على مدار السلسلة. يلتقي هاري بمعظم الشخصيات الرئيسية ويكتسب صديقين مقربين له: رون ويزلي وهو طفلً محبً للمرح من عائلة سحريةٍ عريقةٍ وكبيرةٍ وسعيدةٍ ولكنها فقيرة، وهيرميون غرينجر وهي ساحرةٌ موهوبةٌ ومجتهدةٌ تنتمي إلى عائلة من الماغل (غير السحرة). يواجه هاري أيضًا أستاذ الجرعات في المدرسة، سيفيروس سناب، الذي يظهر كراهيةً عميقةً ودائمةً له بشكلٍ واضحٍ، كما يلتقي هاري بالطفل الشقي والثري دراكو مالفوي الذي يُشَكِّل معه علاقة عداء ومنافسة، ويلتقي هاري أيضًا بمعلم الدفاع ضد فنون الظلام كويرينوس كويرل، الذي يتبين لاحقًا أنه مُتحالِف مع اللورد فولدمورت. يكتشف هاري أيضًا موهبته في الطيران على المكانس ويتم اختياره للعب في فريق جريفندور للكويدتش، وهي رياضةٌ في عالم السحرة حيث يطير اللاعبون على المكانس. يُختتم الكتاب الأول بمواجهة هاري الثانية مع اللورد فولدمورت، الذي يحاول الحصول على جسد ليسترجع قوته من جديد باستخدام قوة حجر الفلاسفة وهي حجر تمنح الحياة الأبدية وتحول أي معدن إلى ذهب خالص.
    
    تستمر السلسلة مع الرواية الثانية هاري بوتر وحجرة الأسرار، التي تحكي عن أحداث السنة الثانية لهاري في هوجورتس. يحقِّق هو وأصدقاؤه في لغز عمره 50 سنة يبدو أنه مرتبط بشكل غريب بالأحداث الخطيرة التي تَقَعُ مُؤَخَّرًا في المدرسة. تبدأ أخت رون الصغرى جيني ويزلي بدراسة سنتها الأولى في هوجورتس، وتجد دفترَ ذكرياتٍ قديمًا بين أدواتها والذي تَبَيَّنَ أنه دفتر ذكريات طالب سابق في هوجورتس اسمه توم مارفولو ريدل، كتبها خلال الحرب العالمية الثانية. تم لاحقًا اكتشاف أنه اللورد فولدمورت عندما كان صغيرًا، وقد سَحَرَ هذا الدفتر بهدف تخليص المدرسة من أصحاب «الدماء الموحلة»، وهو مصطلح مهين يصف السحرة المنحدرين من عائلات الماغل (غير السحرة). تتواجد ذكرى توم ريدل داخل المذكرات وعندما تبدأ جيني بالثقة في دفترَ الذكرياتٍ، يستطيع فولدمورت السيطرة عليها.
    
    من خلال المذكرات، تعمل جيني بناءً على أوامر فولدمورت وتفتح دون وعي «غرفة الأسرار»، مطلقةً بذلك الوحش القديم الذي يسكن الغرفة والذي يبدأ في مهاجمة الطلاب في هوجورتس، تم لاحقًا اكتشاف أن الوحش هو باسيليسك. حيث يقتل كل مَنْ يُجْرِي معه اتصالًا مباشرًا بالعين دون واسطة ويصيب أولئك الذين ينظرون إليه بشكل غير مباشر بحالة من الجمود. يُقدِّم الكتاب أيضًا مُدرِّسًا جديدًا للدفاع ضد فنون الظلام هو غيلدروي لوكهارت، وهو ساحر مبتهج للغاية ومغرور بنفسه ويحب الشهرة والأضواء يتَبَيَّن لاحقًا أنه مجرد محتال.
    
    يكتشف هاري أيضًا قُدْرَتَه على التحدث بلغة الثعابين وهي مهارةٌ نادرةٌ وغالبًا ما ترتبط بـ السحر الأسود. بعد أن تَتَعَرَّضُ هيرميون للهجوم وتتجمد، يتمكن هاري ورون أخيرًا من حل اللغز وفتح غرفة الأسرار، حيث قام هاري بتدمير دفتر الذكريات من أجل إنقاذ جيني، وسيعرف بعد بضع سنوات أنه قام أيضًا بتدمير جزء من روح فولدمورت كان قد خبأه سيد الظلام في الدفتر. تَكْشِف نهاية الكتاب أن لوسيوس مالفوي، والد دراكو هو المجرم الذي وضع دفتر ذكريات اللورد فولدمورت بين أدوات جيني ويزلي.
    
    الرواية الثالثة، هاري بوتر وسجين أزكابان، تحكي عن هاري في سنته الثالثة بمدرسة هوجورتس. وهي الرواية الوحيدة في السلسلة التي لا يظهر فيها اللورد فولدمورت بأي شكلٍ من الأشكال، حيث يتم ذكره فقط. في هذا الكتاب، يجب على هاري أن يتعامل مع تهديد سيريوس بلاك وهو سجينٌ هرب من سجن أزكابان للسحرة بهدف العثور على هاري وقتله. كان بلاك أفضل صديقٍ لوالد هاري، ووفقًا لـ مجتمع السحرة فهو قاتلٌ هاربٌ ساعد في قتل والديْ هاري. يعاني هاري من خوفه الأكبر في هذه الفترة وهو مواجهة الدمنتورات وهي مخلوقاتٌ مظلمةٌ لديها القدرة على التهام الروح البشرية، وتتغذى على الذكريات السعيدة، وتنشر اليأس، وهي مكلفةٌ بحماية المدرسة من سيريوس بلاك. يتعرف هاري على ريموس لوبين، مدرس الدفاع ضد فنون الظلام الذي يكتشف هاري في النهاية أنه مستذئب. يُعَلِّم لوبين هاري تعويذات دفاعية أعلى بكثير من المستوى السحري للأشخاص في سنه.
    
    في النهاية يَعرِف هاري أن كلاًّ من لوبين وبلاك كانا صديقين حميمين لوالده وأن بلاك بريء وقد تم اتهامه من قِبَل صديقهم الرابع بيتر بيتيغرو الذي كان يختبئ في شكل جرذ رون الأليف سكابرز لأنه هو الخائن الحقيقي الذي ساعد فولدمورت في قتل والديْ هاري. في هذا الكتاب، تم التأكيد على موضوع متكرر في السلسلة وهو ظهور مدرس جديد للدفاع ضد فنون الظلام في كل كتاب، ولا يدوم أي منهم أكثر من عام دراسي واحد.
    '''

    # src = 'في النهاية يَعرِف هاري أن كلاًّ من لوبين وبلاك كانا صديقين حميمين لوالده وأن بلاك بريء وقد تم اتهامه من قِبَل صديقهم الرابع بيتر بيتيغرو الذي كان يختبئ في شكل جرذ رون الأليف سكابرز لأنه هو الخائن الحقيقي الذي ساعد فولدمورت في قتل والديْ هاري.'

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ar-he')
    chunker = Chunker(tokenizer=tokenizer, chunk_size=60)
    total_chunks, paragraphs_counter = chunker.split_text2chunks(src)
    print(paragraphs_counter)


if __name__ == "__main__":
    main()