/** The record. Every entry here is a fact taken from Roman's CV (CV___EN v4)
 * or the previous site; nothing is added, inflated, or invented. */

export const profile = {
  /** What he is, not where he works — this line outlives any employer. The
   * Panasonic title lives in the experience list, correctly dated. */
  title: 'AI researcher, ML and distributed systems engineer',
  location: 'Wrocław, Poland',
  /** Corrected directly by Roman: 23 years writing software, 12 of them in
   * AI/ML. The CV's "14+" counted only the professional ML span. */
  yearsSoftware: 23,
  yearsAI: 12,
  summary:
    'I build production AI/ML and data-intensive systems across agentic AI, NLP, semantic search, graph processing and distributed backend platforms — and the high-throughput, horizontally scalable infrastructure they run on. I have been writing software for 23 years — it started with Delphi and Object Pascal for local companies — and working in AI and machine learning for more than 12. I combine that with staff-level architecture ownership, CTO and founder execution, nine publications and eight research and R&D projects, and I specialise in turning research and prototypes into reliable production systems with explicit trade-offs across quality, latency, throughput, cost and operational risk.',
};

export const strengths = [
  {
    k: 'Research to production',
    v: 'I turn AI/ML ideas into evaluated, reproducible and deployable systems, from problem definition and prototyping through production rollout and iteration.',
  },
  {
    k: 'Architecture, performance and reliability',
    v: 'I own critical system design, identify bottlenecks and failure modes, and translate findings into concrete engineering improvements.',
  },
  {
    k: 'Technical leadership and founder execution',
    v: 'I lead design and code reviews, mentor engineers and define technical standards, and have owned product/technology fit, grant writing, hiring and customer-facing delivery as a CTO and co-founder.',
  },
];

type Job = {
  role: string;
  org: string;
  where?: string;
  from?: string;
  to: string;
  /** Set when the role ran alongside a main one, so the list can say so
   * rather than implying two full-time jobs at once. */
  tag?: string;
  notes: string[];
};

export const experience: Job[] = [
  {
    role: 'Looking for the next thing',
    org: 'Independent research',
    where: 'Wrocław',
    from: '09.2026',
    to: 'Present',
    notes: [
      'Experimenting with vector indexes, JEPA-like models and similarity learning.',
    ],
  },
  {
    role: 'Staff Machine Learning Engineer / AI Platform Architect',
    org: 'Panasonic, Global Applied AI',
    where: 'Remote',
    from: '06.2024',
    to: '08.2026',
    notes: [
      "Architected and built the core Agent SDK for KanpAI, Panasonic's enterprise agentic AI platform, defining the agent packaging model, runtime contracts, secure deployment flow and developer experience for BYOA/BYOM use cases.",
      'Designed and built backend services for agent deployment, orchestration and lifecycle management across multi-tenant environments, enabling consistent execution across cloud, on-prem, edge and disconnected contexts.',
      'Created the Agent CLI for packaging, validating and securely deploying agents, including dependency checks, tenant and environment configuration, secrets handling and release safeguards.',
      'Designed massively parallel, at-least-once processing for multi-agent invocation workflows, with in-order processing guarantees where required.',
      'Built observability and evaluation for agentic workflows: tenant-scoped monitoring, model latency and cost tracking, API/SDK evaluation hooks and LLM-as-a-judge regression workflows.',
      'Helped scale KanpAI to 20+ agents across 10+ domains, 20+ model families and 30+ training datasets, supporting 70+ concurrent enterprise users during live workshops and hackathons.',
    ],
  },
  {
    role: 'Staff Engineer',
    org: 'StashAway',
    where: 'Remote',
    from: '10.2020',
    to: '06.2024',
    notes: [
      'Redesigned investment processing pipelines serving hundreds of thousands of customers, simplifying data flow and execution strategy to make end-to-end runs more than 2× faster.',
      'Optimised portfolio rebalancing algorithms and batch execution paths, making rebalancing pipelines 2× faster while preserving correctness, auditability and operational controls.',
      'Owned staff-level architecture for JVM-based investment and platform services, translating broad product and operational requirements into APIs, migration plans and delivery milestones.',
      'Improved reliability of critical fintech workflows through stronger test coverage, observability, rollback safety and incident follow-up.',
    ],
  },
  {
    role: 'Founder / Researcher',
    org: 'Causality · FN · Roosters',
    where: 'Remote · Wrocław',
    from: '10.2020',
    to: 'Present',
    tag: 'Alongside full-time work',
    notes: [
      'Define R&D direction for ML systems, with emphasis on causal reasoning, semantic search, synthetic data generation, graph processing and decision-support tooling.',
      'Write research and grant scopes, convert hypotheses into implementation milestones, and validate prototypes against concrete product and technical risks.',
      'Stay hands-on with architecture, model evaluation, experimental software and technical communication with partners and stakeholders.',
    ],
  },
  {
    role: 'Staff Full-stack Developer, AI Engineer, Freelancer',
    org: 'Toptal',
    where: 'Remote',
    from: '03.2018',
    to: 'Present',
    notes: ['Member of the top-3% global developer network.'],
  },
  {
    role: 'CTO, Co-founder',
    org: 'IamBot (Techstars company)',
    where: 'Wrocław',
    from: '05.2017',
    to: '09.2020',
    notes: [
      'Co-founded the company and owned the technical roadmap for conversational-AI products, from prototype to customer-facing deployments.',
      'Built the core NLP and dialogue stack, the integration layer and the deployment process, while hiring, mentoring and leading the engineering function.',
      'Translated customer discovery into product experiments, prioritising features that improved delivery speed, bot reliability and measurable customer value.',
    ],
  },
  {
    role: 'Research Assistant',
    org: 'Wrocław University of Science and Technology',
    where: 'Wrocław',
    from: '08.2015',
    to: '09.2016',
    notes: [
      'Researched distributed processing of networked data and NLP methods, contributing to publications, grant work and experimental software.',
    ],
  },
  {
    role: 'Software Engineer / Software Architect',
    org: 'Nokia Networks',
    where: 'Wrocław',
    from: '07.2013',
    to: '07.2015',
    notes: [
      'Designed and implemented telecom software components, progressing from software engineering to architecture-level responsibility while improving maintainability, testing and code ownership.',
    ],
  },
  {
    role: 'Web Developer',
    org: 'Chop-Chop.org',
    where: 'Wrocław',
    from: '05.2012',
    to: '07.2013',
    notes: [
      'Delivered client-facing web applications and built foundations in frontend and backend delivery, code quality and customer communication.',
    ],
  },
];

/** Nine, matching the CV. SparklingGraph's documentation entry on Scholar is
 * documentation rather than a publication and lives under Software instead. */
export const publications = [
  {
    title: 'WordNet2Vec: Corpora agnostic word vectorization method',
    venue: 'Neurocomputing 326, 141–150',
    year: '2019',
    authors: 'R Bartusiak, Ł Augustyniak, T Kajdanowicz, P Kazienko, M Piasecki',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:2osOgNQ5qMEC',
  },
  {
    title:
      'This is the way: designing and compiling LEPISZCZE, a comprehensive NLP benchmark for Polish',
    venue: 'Advances in Neural Information Processing Systems 35, 21805–21818',
    year: '2022',
    authors: 'L Augustyniak, K Tagowski, A Sawczyn, D Janiak, R Bartusiak, …',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:nb7KW1ujOQ8C',
  },
  {
    title: 'Choose a job you love: predicting choices of GitHub developers',
    venue: 'IEEE/WIC/ACM International Conference on Web Intelligence, 200–207',
    year: '2016',
    authors: 'R Nielek, O Jarczyk, K Pawlak, L Bukowski, R Bartusiak, A Wierzbicki',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:qjMakFHDy7sC',
  },
  {
    title: 'Sentiment analysis for Polish using transfer learning approach',
    venue: 'Second European Network Intelligence Conference, 53–59',
    year: '2015',
    authors: 'R Bartusiak, L Augustyniak, T Kajdanowicz, P Kazienko',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:u-x6o8ySG0sC',
  },
  {
    title: 'Cooperation prediction in GitHub developers network with restricted Boltzmann machine',
    venue: 'Asian Conference on Intelligent Information and Database Systems, 96–107',
    year: '2016',
    authors: 'R Bartusiak, T Kajdanowicz, A Wierzbicki, L Bukowski, O Jarczyk, …',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:d1gkVwhDpl0C',
  },
  {
    title: 'On quality assessment in Wikipedia articles based on Markov random fields',
    venue: 'Asian Conference on Intelligent Information and Database Systems, 782–791',
    year: '2017',
    authors: 'R Kleminski, T Kajdanowicz, R Bartusiak, P Kazienko',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:CHSYGLWDkRkC',
  },
  {
    title: 'Sentiment analysis based on collaborative data for Polish language',
    venue: 'International Conference on Cooperative Design, Visualization and Engineering',
    year: '2015',
    authors: 'R Bartusiak, T Kajdanowicz',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:u5HHmVD_uO8C',
  },
  {
    title: 'Distributed Processing of Networked Data',
    venue: 'Encyclopedia of Social Network Analysis and Mining, 662–668',
    year: '2018',
    authors: 'P Kazienko, W Indyk, T Kajdanowicz, R Bartusiak',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:dshw04ExmUIC',
  },
  {
    title: 'Distributed Processing of Networked Data',
    venue: 'Encyclopedia of Social Network Analysis and Mining, 1–7',
    year: '2017',
    authors: 'P Kazienko, W Indyk, T Kajdanowicz, R Bartusiak',
    href: 'https://scholar.google.com/citations?view_op=view_citation&hl=en&user=mk68epwAAAJ&citation_for_view=mk68epwAAAJ:UxriW0iASnsC',
  },
];

export const grants = {
  count: 8,
  note: 'Eight research and R&D projects. R&D leader and author for synthetic data generation, semantic search and dialogue-system projects; participant in RENOIR (H2020-MSCA-RISE) and ENGINE (FP7).',
};

type Project = {
  name: string;
  years: string;
  note: string;
  href: string;
  /** Still being built. Labelled in the list so it never reads as something
   * you can pick up and ship with today. */
  wip?: boolean;
};

export const software: Project[] = [
  {
    name: 'BORSUK',
    years: '',
    note: 'Blob-oriented Retrieval with Segmental Unified KNN. Keeps the whole index as immutable Parquet objects in the S3, MinIO, SeaweedFS, GCS or Azure storage you already pay for, and answers a query with a few hundred bytes of resident memory — no always-on RAM cluster to provision, scale or feed between queries.',
    href: 'https://causality.pl/borsuk/',
    wip: true,
  },
  {
    name: 'SFORA',
    years: '',
    note: 'Deep metric learning library. Its method, HERD — Hypergraph EMA-teacher Relational Distillation — plus a multi-model ensemble beat the reported state of the art on the same architecture for CUB-200 by more than a point.',
    href: 'https://causality.pl/sfora/',
    wip: true,
  },
  {
    name: 'SparklingGraph',
    years: '2016 – present',
    note: 'Distributed graph processing library — a Spark GraphX wrapper providing algorithms and APIs. Connects research on networked data to a practical distributed-computing implementation.',
    href: 'http://sparkling.ml',
  },
  {
    name: 'polish-sentiment',
    years: '',
    note: 'Ready-to-use Docker images for sentiment classification of Polish text.',
    href: 'https://github.com/riomus/polish-sentiment',
  },
  {
    name: 'vibe.js',
    years: '',
    note: 'JavaScript implementation of the ViBe background-subtraction algorithm.',
    href: 'https://github.com/riomus/vibe.js',
  },
  {
    name: 'OpticalFlow.js',
    years: '',
    note: 'JavaScript implementation of an optical-flow algorithm.',
    href: 'https://github.com/riomus/OpticalFlow.js',
  },
  {
    name: 'HMM.js',
    years: '',
    note: 'Hidden Markov models in JavaScript, using the Viterbi algorithm.',
    href: 'https://github.com/riomus/HMM.js',
  },
  {
    name: 'Data Projector',
    years: '2015 – present',
    note: 'Java web application for data visualisation and dimensionality change.',
    href: '',
  },
];

export const skills = [
  {
    k: 'AI / ML',
    v: 'Agentic AI, NLP, semantic search, synthetic data, graph ML, representation learning, model evaluation, LLM evaluation',
  },
  {
    k: 'Systems',
    v: 'Distributed systems, event-driven systems, multi-tenant architectures, high-throughput processing, backend platform design',
  },
  {
    k: 'Programming',
    v: 'Python, Kotlin, Java, Scala, Rust, C++, JavaScript/Node.js, Bash',
  },
  {
    k: 'Platforms / tooling',
    v: 'Docker, ArgoCD, Helm, RDS, Spark, Akka, Micronaut, Spring, GitLab CI, Jenkins, Git',
  },
  {
    k: 'Engineering',
    v: 'System architecture, performance profiling, TDD, DDD, functional programming, observability, CI/CD, safe refactoring',
  },
];

/** The original course sites lived on .ml domains that no longer resolve
 * (verified: lsdp.ml and probabilistyczne.ml both fail to connect). Where a
 * repository still carries the material it is linked instead; where none
 * exists the course is listed without a link rather than pointing at a dead
 * host. */
export const courses = [
  {
    name: 'Large-scale data processing',
    href: 'https://github.com/riomus/lsdp',
    hrefLabel: 'riomus/lsdp',
  },
  {
    name: 'Probabilistic machine learning',
    href: 'https://github.com/pwr-ai-pgm/probabilistic.github.com',
    hrefLabel: 'pwr-ai-pgm/probabilistic.github.com',
  },
  {
    name: 'Distributed systems',
    href: 'https://github.com/distributed-systems-wrust/distributed.github.com',
    hrefLabel: 'distributed-systems-wrust/distributed.github.com',
  },
  {
    name: 'Software development paradigms',
    href: 'https://github.com/riomus/paradygmaty/tree/2016/2017-zima',
    hrefLabel: 'riomus/paradygmaty',
  },
];

export const education = [
  {
    what: 'Doctoral studies in Computer Science',
    org: 'Wrocław University of Science and Technology · Computer Science and Management',
    when: '2015 – 2022',
    note: 'Research area: scalable massive networked data processing. Supervisor: Prof. Przemysław Kazienko.',
  },
  {
    what: 'MSc, Intelligent Information Systems',
    org: 'Wrocław University of Science and Technology · Computer Science and Management',
    when: '2014 – 2015',
    note: 'Grade 5.5 / A+. Thesis: Sentiment Analysis and Opinion Mining in Selected Social Platform.',
  },
  {
    what: 'BSc, Computer Science',
    org: 'Wrocław University of Science and Technology · Computer Science and Management',
    when: '2010 – 2014',
    note: 'Grade 5.5 / A+.',
  },
];

export const certifications = [
  { what: 'Reactive Architecture: Reactive Microservices', org: 'Akka', when: '2020' },
  { what: 'Reactive Architecture: Building Scalable Systems', org: 'Akka', when: '2020' },
];

export const languages = [
  { k: 'Polish', v: 'Native' },
  { k: 'English', v: 'Full professional proficiency' },
  { k: 'German', v: 'Elementary proficiency' },
];
