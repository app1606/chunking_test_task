import pandas as pd
from fixed_token_chunker_modified import FixedTokenChunker
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def range_len(ranges):
  return sum([end_ - start_ for (start_, end_) in ranges])

def intersect(range1, range2):
  start_id = max(range1[0], range2[0])
  end_id = min(range1[1], range2[1])

  if start_id <= end_id:
    return (start_id, end_id)
  else:
    return None

def unite(ranges):
  if ranges == []:
    return None

  ranges = sorted(ranges, key=lambda x: x[0])
  ans = [ranges[0]]

  for current_ in ranges[1:]:
    if ans[-1][1] >= current_[0]:
      ans[-1] = (ans[-1][0], max(ans[-1][1], current_[1]))
    else:
      ans.append(current_)

  return ans


def prec_recall_(relevant_indices, golden_indices):
  questions_number = len(relevant_indices)
  for question_index in range(questions_number): # iterate over questions
    intersection_sections = []

    for chunk in relevant_indices[question_index]:
      for exc in golden_indices[question_index]:
        intersection = intersect(chunk, exc)

        if not isinstance(intersection, type(None)):
          intersection_sections.append(intersection)

    intersection = unite(intersection_sections)

    if isinstance(intersection, type(None)):
      len_intersection = 0
    else:
      len_intersection = range_len(intersection)

    precision = len_intersection / range_len(relevant_indices[question_index])
    recall = len_intersection / range_len(golden_indices[question_index])

  return precision, recall

def get_golden_indices(questions_reference):
  golden_indices = []

  for _, row in questions_reference.iterrows():
    start_indices = [x['start_index'] for x in row['references']]
    end_indices = [x['end_index'] for x in row['references']]
    ranges = list(zip(start_indices, end_indices))
    golden_indices.append(ranges)

  return golden_indices

def evaluate_chunker(corpus, questions_reference, chunker, emb_function, num_retrieved):
  questions = questions_reference['question'].to_list()
  chunks, indices = chunker.split_text(corpus)

  questions_embeddings = emb_function(questions)
  chunks_embeddings = emb_function(chunks)

  similarity = cosine_similarity(questions_embeddings, chunks_embeddings)
  sorted_chunk_indices = (-similarity).argsort(axis=1)[:, :num_retrieved]

  relevant_indices = np.array(indices)[sorted_chunk_indices]

  golden_indices = get_golden_indices(questions_reference)

  precision, recall = prec_recall_(relevant_indices, golden_indices)

  return precision, recall

