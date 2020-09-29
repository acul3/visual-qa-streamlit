FROM jjanzic/docker-python3-opencv

RUN git clone https://github.com/huggingface/transformers.git

WORKDIR transformers/examples/lxmert/

RUN pip install -r requirements.txt

RUN pip install streamlit

COPY model.py model.py
# download model parameters
RUN python model.py

COPY app.py app.py


CMD streamlit run app.py
