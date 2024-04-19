import re
import chromadb
from sentence_transformers import  SentenceTransformer
import streamlit as st



# Initializing chromaDB
client = chromadb.PersistentClient(path="E://search_engine_db") #_test_db
collection = client.get_collection(name="search_engine") #test_collection
collection_name = client.get_collection(name="search_engine_FileName")
model_name="paraphrase-MiniLM-L3-V2"
model = SentenceTransformer(model_name, device="cuda")

def clean_data(data): # data is the entire text file entry in the dataframe

    # removing timestamps
    data = re.sub("\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}"," ",  data)

    # removing index no. of dialogues
    data = re.sub(r'\n?\d+\r', "", data)

    # removing escape sequences like \n \r
    data = re.sub('\r|\n', "", data)

    # removing <i> and </i>
    data = re.sub('<i>|</i>', "", data)
    # removing links
    data = re.sub("(?:www\.)osdb\.link\/[\w\d]+|www\.OpenSubtitles\.org|osdb\.link\/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com", " ",data)

    # Converting to lower case
    data = data.lower()

    # Join and return
    return data

def extract_id(id_list):
    new_id_list=[]
    for item in id_list:
        match = re.match(r'^(\d+)', item)
        if match:
            extracted_number = match.group(1)
            new_id_list.append(extracted_number)
    return new_id_list

st.header("Movie Subtitle Search Engine") 
search_query=st.text_input("Enter a dialogue to search....")
if st.button("Search")==True:
    
    st.subheader("Relevant Subtitle Files")
    search_query=clean_data(search_query)
    query_embed = model.encode(search_query).tolist()

    search_results=collection.query(query_embeddings=query_embed, n_results=10)
    id_list = search_results['ids'][0]

    id_list = extract_id(id_list)
    print(id_list)
    for id in id_list:
        file_name = collection_name.get(ids=f"{id}")["documents"][0]
        st.markdown(f"[{file_name}](https://www.opensubtitles.org/en/subtitles/{id})")



