//! UTC civil dates without a date crate: Howard Hinnant's `civil_from_days`
//! and `days_from_civil`, exact for the proleptic Gregorian calendar.

/// `(year, month, day)` for `days` since 1970-01-01.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the algorithm bounds the day to 1..=31 and the month to 1..=12"
)]
pub(crate) fn from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(month <= 2), month, day)
}

/// Days since 1970-01-01 for a civil date; `None` when the month or day is
/// out of range.
pub(crate) fn to_days(year: i64, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400); // [0, 399]
    let mp = i64::from(month) + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    Some(era * 146_097 + doe - 719_468)
}

/// `secs` after the Unix epoch as RFC 3339 in UTC (`2026-10-01T13:02:22Z`).
pub(crate) fn rfc3339(secs: u64) -> String {
    let (y, mo, d) = from_days(i64::try_from(secs / 86_400).unwrap_or(i64::MAX));
    let t = secs % 86_400;
    let (h, m, s) = (t / 3_600, t % 3_600 / 60, t % 60);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_across_leap_and_century_boundaries() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(4_107_542_399), "2100-02-28T23:59:59Z");
        assert_eq!(rfc3339(4_107_542_400), "2100-03-01T00:00:00Z");
        assert_eq!(rfc3339(1_790_859_742), "2026-10-01T13:02:22Z");
    }

    #[test]
    fn to_days_inverts_from_days() {
        for days in [-719_468, -1, 0, 1, 11_016, 20_634, 47_540, 2_932_896] {
            let (y, m, d) = from_days(days);
            assert_eq!(to_days(y, m, d), Some(days), "{y}-{m}-{d}");
        }
        assert_eq!(to_days(2026, 13, 1), None);
        assert_eq!(to_days(2026, 2, 0), None);
    }
}
