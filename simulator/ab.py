"""First-session customer-level two-proportion Z test (not impression CTR inference)."""
import math


def wilson(successes, n):
    if type(n) is not int or type(successes) is not int or n <= 0 or not 0 <= successes <= n:
        raise ValueError('Require integer 0 <= successes <= n and n > 0')
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return [center-radius, center+radius]


def compare_customers(records, min_per_arm=100):
    """Each customer has exactly one fixed A/B assignment and one binary outcome.

    Caller must provide whether each customer clicked during their first session.
    Replayed models for the same customers are rejected (paired, not this test).
    """
    if type(min_per_arm) is not int or min_per_arm < 1:
        raise ValueError('min_per_arm must be positive')
    arms = {'A': [], 'B': []}
    seen = set()
    for r in records:
        cid, arm, outcome = r['customer_id'], r['variant'], r['clicked_first_session']
        if not isinstance(cid, str) or not cid or cid in seen or arm not in arms or type(outcome) is not bool:
            raise ValueError('Unique customer, A/B variant and boolean clicked_first_session required')
        seen.add(cid)
        arms[arm].append(outcome)
    if not all(arms.values()):
        raise ValueError('Both A and B require customers')
    n1,n2 = len(arms['A']),len(arms['B'])
    s1,s2 = sum(arms['A']),sum(arms['B'])
    p1,p2 = s1/n1,s2/n2
    l1,u1 = wilson(s1,n1); l2,u2 = wilson(s2,n2)
    # Newcombe independent-proportion difference interval using Wilson intervals.
    diff = p2-p1
    ci = [diff-math.sqrt((p2-l2)**2+(u1-p1)**2), diff+math.sqrt((u2-p2)**2+(p1-l1)**2)]
    pooled=(s1+s2)/(n1+n2)
    expected=[n1*pooled,n1*(1-pooled),n2*pooled,n2*(1-pooled)]
    sufficient = min(n1,n2) >= min_per_arm and min(expected) >= 5
    se=math.sqrt(pooled*(1-pooled)*(1/n1+1/n2))
    z=diff/se if se else None
    return {'metric':'first_session_customer_click_rate','unit':'customer','kind':'synthetic_randomized_first_session',
            'A':{'customers':n1,'clicked_customers':s1,'rate':p1,'ci95': [l1,u1]},
            'B':{'customers':n2,'clicked_customers':s2,'rate':p2,'ci95': [l2,u2]},
            'difference_B_minus_A':diff,'ci95':ci,'z':z if sufficient else None,
            'p_value':math.erfc(abs(z)/math.sqrt(2)) if sufficient and z is not None else None,
            'status':'test_computed' if sufficient else 'insufficient_sample_or_expected_counts',
            'interpretation':'Unadjusted per-persona first-session test; synthetic outcomes only; no automatic winner; no repeated peeking'}
