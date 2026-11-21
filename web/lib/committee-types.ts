export type CommitteeAlternative = {
  members: string[];
  mentor: null | string;
  score: number;
  supporting_diploma_ids: string[];
};

export type CommitteePerson = {
  buddy_score: number;
  defense_score: number;
  expertise_score: number;
  name: string;
  prior_score: number;
  score: null | number;
  supporting_diploma_ids: string[];
  supporting_paper_titles: string[];
};

export type CommitteeRequest = {
  abstract?: string;
  alternatives?: number;
  include_prior: false;
  keywords?: string[];
  mentor?: string;
  research_area?: string;
  study_program?: string;
  title: string;
};

export type CommitteeResponse = {
  alternatives: CommitteeAlternative[];
  confidence: number;
  confidence_reasons: string[];
  evidence: {
    similar_diplomas: SimilarDiploma[];
    supporting_paper_titles: string[];
  };
  members: CommitteePerson[];
  mentor: CommitteePerson;
  mentor_is_given: boolean;
  mode: 'full' | 'members_only';
  supporting_diploma_ids: string[];
};

export type Professor = {
  cabinet: null | string;
  consultations: null | string;
  id: null | number;
  mentor_eligible: boolean;
  name: string;
  photo: null | string;
  position: null | string;
  profile: null | string;
  title: null | string;
};

export type SimilarDiploma = {
  external_id: string;
  id: string;
  members: string[];
  mentor: string;
  rerank_score: null | number;
  similarity: number;
  title: string;
};
