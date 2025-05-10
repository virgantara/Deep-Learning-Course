import torch
from model import JenisKelaminClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("jenis_kelamin.csv")
tb_min, tb_max = train_df["TB"].min(), train_df["TB"].max()
bb_min, bb_max = train_df["BB"].min(), train_df["BB"].max()


infer_df = pd.read_csv("jenis_kelamin_infer.csv")
infer_df['TB'] = (infer_df['TB'] - tb_min) / (tb_max - tb_min)
infer_df['BB'] = (infer_df['BB'] - bb_min) / (bb_max - bb_min)
x_infer = torch.tensor(infer_df[['TB', 'BB']].values, dtype=torch.float32)

model = JenisKelaminClassifier(8, 2)
model.load_state_dict(torch.load('jenis_kelamin_model.pth'))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    logits = model(x_infer)
    preds = logits.argmax(dim=1)

le = LabelEncoder()
le.fit(["Perempuan", "Laki-laki"])
predicted_labels = le.inverse_transform(preds.numpy())

infer_df['Prediksi_Jenis_Kelamin'] = predicted_labels
print(infer_df)

infer_df.to_csv("hasil_inference.csv", index=False)