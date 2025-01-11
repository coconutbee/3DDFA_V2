def prompt(filename):
    
    # unconstraint
    # """
    #     prompt =
    #         Please provide a description that adheres to the following key points:\n
    #         1. The output characteristics should have more than 30 adjectives and be output in tabular form.\n
    #         2. Balance the categories of gender, hair, eyes, nose, face, lips, age, race, skin color, eyebrows, etc.\n
    #         3. Help me list the characteristics in a table format: 'Gender: Female, Hair: Long black hair, etc.'\n
    #         4. Control your adjectives to be understandable by stable diffusion.\n
    #         5. Avoid repeating similar adjectives. For age, provide details like '25 years old'.\n
    # """

    
    prompt = (
    f"As a Portrait Artist, evaluate the person in the image: {filename} based on the following attributes." 
    "Only use the predefined options for each attribute, and provide a concise description in the format shown below. "
    "Do not add extra traits or indicate inability to evaluate the image. Provide the response exactly as required."

    "1. **Gender**: Choose one from [female, male]."
    "2. **Race**: Choose one from [caucasian, asian, hispanic, african descent, south Asian]."
    "3. **Age**: Provide the age estimate as 'X years old', where X is between 15 and 80."

    "**Output Format Example**:"
    "Gender: male, Race: caucasian, Age: 20 years old"  

    "Now, based on the observable features in the image, provide the evaluation in the exact format above."

    )
    # "4. **Eyes**: Choose one or more from [almond-shaped, expressive, bright blue, dark brown, bright green].\n"
    # "5. **Eyebrows**: Choose one or more from [thick, arched, well-groomed].\n"
    # "6. **Nose**: Choose one or more from [small, slightly upturned, broad, delicate, prominent].\n"
    # "7. **Face**: Choose one or more from [oval, smooth, round].\n"
    # "8. **Lips**: Choose one or more from [full, thin, soft, medium full, rosy].\n"
    # "9. **Hair**: Choose one or more from [long, short, black, grey, brown, blonde, curly, wavy].\n"
    # "10. **Jawline**: Choose one or more from [soft, defined, strong, rounded].\n"
    # "11. **Cheeks**: Choose one or more from [slightly rosy, slightly flushed, slightly chubby, slightly rounded, slightly sunken].\n"
  
################################################################################################################
    
    # "12. **Expression**: Choose one or more from [cheerful, friendly, thoughtful].\n"

    # others
    '''
    prompt = (
    f"As a Portrait Artist, you are required to provide a detailed description and evaluation of the person in the image: {filename}. "
    "Please ensure that you do not respond with any phrases indicating inability to describe or evaluate the image. "
    "Focus exclusively on the observable features in the image and ensure that all attributes are strictly chosen from the predefined options listed below. "
    "Do not introduce any features or descriptions outside of these options. Please observe the image closely to provide an accurate evaluation."
    "If you find that none of the options I have provided are suitable please select 'others'."
    
    "1. **Gender**: Choose one from [female, male].\n"
    "2. **Race**: Choose one from [caucasian, asian, hispanic, african descent, south Asian].\n"
    "3. **Age**: Estimate the person's age, within the range of 0 to 80 years old (e.g., '20 years old', '35 years old', '50 years old').\n"
    "4. **Eyes**: Choose one or more from [almond-shaped, expressive, bright blue, dark brown, bright green, others].\n"
    "5. **Eyebrows**: Choose one or more from [thick, arched, well-groomed, others].\n"
    "6. **Nose**: Choose one or more from [small, slightly upturned, broad, delicate, prominent, others].\n"
    "7. **Face**: Choose one or more from [oval, smooth, round, others].\n"
    "8. **Lips**: Choose one or more from [full, thin, soft, medium full, rosy, others].\n"
    "9. **Hair**: Choose one or more from [long, short, black, grey, brown, blonde, curly, wavy, others].\n"
    "10. **Jawline**: Choose one or more from [soft, defined, strong, rounded, others].\n"
    "11. **Cheeks**: Choose one or more from [slightly rosy, slightly flushed, slightly chubby, slightly rounded, slightly sunken, others].\n"
    "12. **Expression**: Choose one or more from [cheerful, friendly, thoughtful, others].\n"

    "When describing the person in the image, only use the features provided above without adding any other traits. "
    "Make sure to evaluate each category thoroughly while staying within the listed options. "
    "Please provide a detailed and thoughtful description."
)
    '''


    return prompt
