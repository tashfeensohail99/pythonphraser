"""Immigration document vocabulary.

DOC_TYPES is the closed set the LLM classifier chooses from (must match the
docType tags the backend seeds onto templates). FIELD_KEYS is a fixed superset
of extractable fields — fixed (not per-type) because OpenAI Structured Outputs
in strict mode requires a closed object schema. Unused fields just come back
null for a given document.
"""

DOC_TYPES: dict[str, dict] = {
    "PASSPORT": {
        "desc": "Passport biographic data page",
        "fields": ["fullName", "passportNumber", "dateOfBirth", "expiryDate", "issueDate", "nationality"],
    },
    "NATIONAL_ID": {
        "desc": "National identity card (e.g. CNIC)",
        "fields": ["fullName", "idNumber", "dateOfBirth", "expiryDate"],
    },
    "BANK_STATEMENT": {
        "desc": "Bank account statement / proof of funds",
        "fields": ["accountHolder", "bankName", "statementDate", "closingBalance", "currency"],
    },
    "LANGUAGE_TEST": {
        "desc": "English/French language test result (IELTS, TOEFL, PTE, Duolingo)",
        "fields": ["fullName", "testName", "overallScore", "issueDate", "expiryDate"],
    },
    "ACADEMIC_TRANSCRIPT": {
        "desc": "Academic transcript / marksheet",
        "fields": ["fullName", "institutionName", "documentDate"],
    },
    "EDUCATION_CERTIFICATE": {
        "desc": "Degree, diploma or graduation certificate",
        "fields": ["fullName", "institutionName", "programOrCourse", "issueDate"],
    },
    "ACCEPTANCE_LETTER": {
        "desc": "Letter of acceptance / admission from an institution",
        "fields": ["fullName", "institutionName", "programOrCourse", "documentDate"],
    },
    "STATEMENT_OF_PURPOSE": {
        "desc": "Statement of purpose / study plan essay",
        "fields": ["fullName", "summary"],
    },
    "RESUME": {
        "desc": "CV / resume",
        "fields": ["fullName", "summary"],
    },
    "POLICE_CLEARANCE": {
        "desc": "Police clearance / good-conduct certificate",
        "fields": ["fullName", "issuingAuthority", "issueDate", "expiryDate"],
    },
    "MEDICAL_EXAM": {
        "desc": "Immigration medical examination result",
        "fields": ["fullName", "issuingAuthority", "issueDate", "expiryDate"],
    },
    "MARRIAGE_CERTIFICATE": {
        "desc": "Marriage certificate",
        "fields": ["fullName", "issuingAuthority", "documentDate"],
    },
    "BIRTH_CERTIFICATE": {
        "desc": "Birth certificate",
        "fields": ["fullName", "dateOfBirth", "issuingAuthority"],
    },
    "FAMILY_REGISTRATION_CERTIFICATE": {
        "desc": (
            "NADRA Family Registration Certificate (FRC) / family registration / "
            "family-tree certificate that lists the members of a family and their "
            "relationships. Often cites several CNIC numbers — it is NOT a single "
            "national ID card."
        ),
        "fields": ["fullName", "issuingAuthority", "documentDate", "summary"],
    },
    "DIVORCE_CERTIFICATE": {
        "desc": "Divorce certificate / decree / deed (e.g. talaq, khula)",
        "fields": ["fullName", "issuingAuthority", "documentDate"],
    },
    "DEATH_CERTIFICATE": {
        "desc": "Death certificate",
        "fields": ["fullName", "issuingAuthority", "documentDate"],
    },
    "EMPLOYMENT_LETTER": {
        "desc": "Employment / offer / experience letter",
        "fields": ["fullName", "employerName", "jobTitle", "documentDate"],
    },
    "LMIA": {
        "desc": "Labour Market Impact Assessment (Canada)",
        "fields": ["employerName", "issuingAuthority", "documentDate"],
    },
    "BUSINESS_PLAN": {
        "desc": "Business plan (e.g. E2 visa)",
        "fields": ["summary", "documentDate"],
    },
    "INCORPORATION": {
        "desc": "Business registration / certificate of incorporation",
        "fields": ["employerName", "issuingAuthority", "issueDate"],
    },
    "TAX_RETURN": {
        "desc": "Tax return / tax assessment document",
        "fields": ["fullName", "documentDate"],
    },
    "SPONSORSHIP_LETTER": {
        "desc": "Sponsorship / affidavit of support letter",
        "fields": ["fullName", "summary", "documentDate"],
    },
    "TRAVEL_ITINERARY": {
        "desc": "Flight / travel itinerary",
        "fields": ["fullName", "documentDate"],
    },
    "VISA": {
        "desc": "Visa sticker / previous visa copy",
        "fields": ["fullName", "issueDate", "expiryDate", "issuingAuthority"],
    },
    "PHOTOGRAPH": {
        "desc": "Passport-style portrait photograph",
        "fields": [],
    },
    "OTHER": {
        "desc": "Any document not matching the categories above",
        "fields": ["summary"],
    },
}

# Fixed superset of every field any type can yield (for the strict schema).
FIELD_KEYS: list[str] = [
    "fullName",
    "passportNumber",
    "idNumber",
    "dateOfBirth",
    "nationality",
    "issueDate",
    "expiryDate",
    "documentDate",
    "statementDate",
    "accountHolder",
    "bankName",
    "closingBalance",
    "currency",
    "institutionName",
    "programOrCourse",
    "testName",
    "overallScore",
    "employerName",
    "jobTitle",
    "issuingAuthority",
    "summary",
]
