"""
A hand-labelled retrieval set: a small company handbook plus queries whose
correct source document is known.

Deliberately small and hand-written. The point is not a benchmark score —
20 documents cannot produce one — but a repeatable measurement of whether
query expansion and reranking help or hurt on this corpus, so the latency
and token cost they add can be justified or dropped.

Queries are phrased the way a user would ask, not by quoting the document,
so lexical overlap alone does not answer them. Several are deliberately
adversarial: `corpus` contains near-duplicate topics (parental vs sick
leave, laptop vs phone policy) that a retriever can confuse.
"""

from dataclasses import dataclass

CORPUS: dict[str, str] = {
    "refunds": (
        "Refund policy. Customers may return any physical item within 30 "
        "days of delivery for a full refund. Refunds are issued to the "
        "original payment method and take 5 business days to appear. "
        "Opened software licences and digital downloads are non-refundable "
        "once the activation key has been revealed."
    ),
    "shipping": (
        "Shipping. Standard delivery is free on orders above fifty euros "
        "and costs 4.90 euros otherwise. Express delivery is 12 euros and "
        "arrives the next business day if ordered before 2pm. We ship to "
        "the EU, the UK and Switzerland. We do not ship batteries by air."
    ),
    "warranty": (
        "Warranty. Hardware carries a 24 month manufacturer warranty from "
        "the date of purchase. The warranty covers manufacturing defects "
        "but not accidental damage, liquid damage, or wear to consumable "
        "parts such as batteries and cables."
    ),
    "parental-leave": (
        "Parental leave. Employees who become parents are entitled to 16 "
        "weeks of paid parental leave, which may be taken in up to three "
        "separate blocks within the first two years. Leave must be "
        "requested at least eight weeks in advance through the HR portal."
    ),
    "sick-leave": (
        "Sick leave. Employees may take up to 10 paid sick days per "
        "calendar year without a medical certificate. Absences longer than "
        "three consecutive days require a doctor's note. Unused sick days "
        "do not carry over to the following year."
    ),
    "holiday": (
        "Annual holiday. Full-time employees accrue 25 days of paid "
        "holiday per year, plus public holidays. Up to five unused days "
        "may be carried into the next year and must be used by March 31."
    ),
    "laptop-policy": (
        "Laptop policy. Every employee receives a company laptop on their "
        "first day. Laptops are replaced every three years, or sooner if "
        "the hardware fails. Personal use is permitted but the device must "
        "be returned when employment ends. Full-disk encryption is "
        "mandatory and enforced by the IT team."
    ),
    "phone-policy": (
        "Mobile phone policy. Employees in customer-facing roles receive a "
        "company mobile phone and a monthly allowance of 30 euros. Other "
        "employees may claim a phone stipend of 15 euros per month instead "
        "of receiving a device."
    ),
    "expenses": (
        "Expense reimbursement. Submit receipts through the finance portal "
        "within 60 days of the expense. Claims under 50 euros do not need "
        "manager approval. Travel booked outside the approved portal is "
        "reimbursed only with prior written approval."
    ),
    "remote-work": (
        "Remote work. Employees may work remotely up to three days per "
        "week. Fully remote arrangements require director approval and a "
        "documented home office setup. Teams must overlap at least four "
        "hours with Central European Time."
    ),
    "onboarding": (
        "Onboarding. New joiners complete a two week onboarding programme "
        "covering security training, the codebase walkthrough and a "
        "shadowing rotation. A buddy is assigned for the first 90 days."
    ),
    "security-incident": (
        "Security incidents. Report any suspected breach to the security "
        "team within one hour of discovery, including lost devices and "
        "suspected phishing. Do not attempt to investigate the incident "
        "yourself. The on-call security engineer is reachable at all hours."
    ),
    "password-policy": (
        "Passwords and access. All accounts require multi-factor "
        "authentication. Passwords must be at least 14 characters and are "
        "stored only in the company password manager. Credentials must "
        "never be shared over chat or email, and service accounts are "
        "rotated every 90 days."
    ),
    "data-retention": (
        "Data retention. Customer support transcripts are retained for 24 "
        "months and then deleted. Invoices are retained for ten years to "
        "meet tax obligations. Deletion requests are honoured within 30 "
        "days except where a legal retention period applies."
    ),
    "support-hours": (
        "Support hours. Customer support is staffed from 8am to 8pm "
        "Central European Time on weekdays. Weekend cover is limited to "
        "urgent incidents raised through the emergency line. First "
        "response targets are four hours on weekdays."
    ),
    "pricing-tiers": (
        "Pricing. The Starter plan is 19 euros per seat per month, Growth "
        "is 49 euros per seat and Enterprise is priced individually. "
        "Annual billing carries a 20 percent discount. Plan changes take "
        "effect at the start of the next billing cycle."
    ),
    "trial": (
        "Free trial. New accounts get a 14 day trial of the Growth plan "
        "with no payment details required. Trials can be extended once by "
        "seven days on request. Data created during a trial is kept if the "
        "account converts within 30 days of the trial ending."
    ),
    "sla": (
        "Service levels. Enterprise customers are covered by a 99.9 "
        "percent monthly uptime commitment. Credits of 10 percent of the "
        "monthly fee apply for each full percentage point below target. "
        "Scheduled maintenance is excluded and announced 72 hours ahead."
    ),
    "api-limits": (
        "API rate limits. The default limit is 600 requests per minute per "
        "API key. Bursts up to 1000 are tolerated for 10 seconds. "
        "Exceeding the limit returns HTTP 429 with a Retry-After header. "
        "Limits can be raised for Enterprise accounts on request."
    ),
    "offboarding": (
        "Offboarding. On an employee's last day, access is revoked within "
        "two hours, hardware is collected, and outstanding expenses are "
        "settled in the next payroll run. Personal data is removed from "
        "company devices before reassignment."
    ),
}


@dataclass(frozen=True)
class Query:
    text: str
    # The document that genuinely answers the question.
    relevant_doc: str


QUERIES: tuple[Query, ...] = (
    Query("how long do I have to send something back?", "refunds"),
    Query("can I get my money back on a downloaded program?", "refunds"),
    Query("when is delivery free?", "shipping"),
    Query("how fast can I get my order if I pay more?", "shipping"),
    Query("is a broken battery covered after a year?", "warranty"),
    Query("I am having a baby, how much time off do I get?", "parental-leave"),
    Query("do I need a doctor's note to stay home ill?", "sick-leave"),
    Query("how many vacation days can I roll over?", "holiday"),
    Query("when does the company give me a new computer?", "laptop-policy"),
    Query("do I get money towards my mobile?", "phone-policy"),
    Query("how late can I claim a taxi receipt?", "expenses"),
    Query("how many days a week can I stay home to work?", "remote-work"),
    Query("how long is the programme for new starters?", "onboarding"),
    Query("I think I clicked a phishing link, what now?", "security-incident"),
    Query("how long should my login secret be?", "password-policy"),
    Query("how long do you keep my chat history?", "data-retention"),
    Query("can someone help me on a Sunday?", "support-hours"),
    Query("what do I pay for the middle plan?", "pricing-tiers"),
    Query("can I try it without giving a card?", "trial"),
    Query("what happens if the service is down too often?", "sla"),
    Query("how many calls per minute am I allowed?", "api-limits"),
    Query("when is my account closed after I leave?", "offboarding"),
)
