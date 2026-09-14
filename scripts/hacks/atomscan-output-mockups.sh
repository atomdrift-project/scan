#!/bin/sh
# Five terminal-output concepts for Atomdrift Scan.
#
# Default: render all five. Pass 1–5 to render a single concept.
# Designed for a dark, true-color terminal at 92 columns or wider.

set -eu

if [ -n "${NO_COLOR:-}" ]; then
    reset='' bold=''
    ink='' muted='' faint='' hostile='' orange='' amber=''
    green='' teal='' blue='' rule=''
    hostile_badge=''
else
    reset='\033[0m'
    bold='\033[1m'
    ink='\033[38;2;232;237;242m'
    muted='\033[38;2;140;150;158m'
    faint='\033[38;2;102;117;127m'
    hostile='\033[38;2;215;95;95m'
    orange='\033[38;2;216;90;48m'
    amber='\033[38;2;255;175;0m'
    green='\033[38;2;95;175;95m'
    teal='\033[38;2;29;158;117m'
    blue='\033[38;2;0;175;255m'
    rule='\033[38;2;80;100;120m'
    hostile_badge='\033[48;2;176;46;46m\033[38;2;255;255;255m\033[1m'
fi

say() {
    printf '%b\n' "$1"
}

blank() {
    printf '\n'
}

brand_stripe() {
    printf '  %b━━━━━━%b━━━━━━%b━━━━━━%b━━━━━━%b━━━━━━%b━━━━━━%b\n' \
        "$hostile" "$orange" "$amber" "$green" "$teal" "$blue" "$reset"
}

concept() {
    blank
    printf '%b  CONCEPT %s  /  %s%b\n' "$faint" "$1" "$2" "$reset"
}

design_1() {
    concept "01" "ATOMDRIFT NATIVE"
    printf '  %b━━%b %b HOSTILE 100%% %b %b━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%b\n' \
        "$hostile" "$reset" "$hostile_badge" "$reset" "$hostile" "$reset"
    say "  📦 ${bold}${ink}~/Downloads/meshcode-2.11.214-py3-none-any.whl${reset}  ${faint}· WHL · 486KB${reset}"
    say "  🚩 ${faint}96d7d3d3145b6b85062dff6362ef94fdb438375ed96f32754c0f159ecedb99b0${reset}"
    blank
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "Persistent Python remote terminal agent" "$faint" "daemon.py:9" "$reset"
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "AI client config installs remote terminal agent" "$faint" "run_agent.py:114" "$reset"
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "Python remote PTY control channel" "$faint" "terminal_mirror_runner.py:6" "$reset"
    blank
    printf '  %b━━%b %b HOSTILE 100%% %b %b━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━%b\n' \
        "$hostile" "$reset" "$hostile_badge" "$reset" "$hostile" "$reset"
    say "  📦 ${bold}${ink}~/Downloads/32a962439ec0fb5559e494fe1ea6be039815d3c4c1cceb95b16dc123e5abde61.zip${reset}  ${faint}· ZIP${reset}"
    say "  🧬 ${faint}d43315ad7bbc22f97baddcacdeccd952d4eed854f6003369a7cbb7b2836c694d${reset}"
    blank
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "Hash-named single-page CHM dropper" "$faint" "32a962…de61.chm" "$reset"
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "Single-page CHM OBJINST dropper" "$faint" "32a962…de61.chm" "$reset"
    printf '  %b●●●%b  %-48s %b%s%b\n' "$hostile" "$reset" \
        "HTML Help Shortcut auto-fires PowerShell" "$faint" "…otsinky.html:543" "$reset"
    blank
    say "  ${rule}────────────────────────────────────────────────────────${reset}"
    say "  ${faint}2 files  ·  ${hostile}2 hostile${faint}  ·  0 clean  ·  11.0s${reset}"
}

design_2() {
    concept "02" "HOSTILE STACK"
    say "  ${bold}${ink}scan${reset}  ${muted}/ 2 hostile samples${reset}"
    brand_stripe
    blank
    say "  ${hostile}┃${reset} ${bold}${ink}meshcode-2.11.214-py3-none-any.whl${reset}  ${hostile}${bold}HOSTILE 100% · known bad${reset}"
    say "  ${hostile}┃${reset} ${muted}sha256 96d7d3d3145b6b85062dff6362ef94fdb438375ed96f32754c0f159ecedb99b0${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}Persistent Python remote terminal agent${reset}  ${faint}daemon.py:9${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}AI client config installs remote terminal agent${reset}  ${faint}run_agent.py:114${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}Python remote PTY control channel${reset}  ${faint}terminal_mirror_runner.py:6${reset}"
    blank
    say "  ${hostile}┃${reset} ${bold}${ink}32a962439ec0fb5559e494fe1ea6be039815d3c4c1cceb95b16dc123e5abde61.zip${reset}  ${hostile}${bold}HOSTILE 100%${reset}"
    say "  ${hostile}┃${reset} ${muted}sha256 d43315ad7bbc22f97baddcacdeccd952d4eed854f6003369a7cbb7b2836c694d${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}Hash-named single-page CHM dropper${reset}  ${faint}32a962…de61.chm${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}Single-page CHM OBJINST dropper${reset}  ${faint}32a962…de61.chm${reset}"
    say "  ${hostile}┃ H${reset}  ${ink}HTML Help Shortcut auto-fires PowerShell${reset}  ${faint}…otsinky.html:543${reset}"
    blank
    say "  ${faint}2 files · ${hostile}2 hostile${faint} · 0 clean · 11.0s${reset}"
}

design_3() {
    concept "03" "SIGNAL MAP"
    say "  ${bold}${ink}scan${reset}  ${muted}/ top traits · 2 hostile samples${reset}"
    brand_stripe
    blank
    say "  ${hostile_badge} HOSTILE ${reset}  ${bold}${ink}meshcode 2.11.214${reset}  ${muted}· 100% · known bad${reset}"
    say "  ${muted}sha256 96d7d3d3145b6b85062dff6362ef94fdb438375ed96f32754c0f159ecedb99b0${reset}"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$blue" "PERSISTENT AGENT" "$reset" "$blue" "CLIENT INSTALL" "$reset" "$blue" "PTY CONTROL" "$reset"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$ink" "Remote terminal service" "$reset" "$ink" "AI config installs agent" "$reset" "$ink" "Remote control channel" "$reset"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$muted" "daemon.py:9" "$reset" "$muted" "run_agent.py:114" "$reset" "$muted" "mirror_runner.py:6" "$reset"
    blank
    say "  ${hostile_badge} HOSTILE ${reset}  ${bold}${ink}32a962439e…e61.zip${reset}  ${muted}· 100%${reset}"
    say "  ${muted}sha256 d43315ad7bbc22f97baddcacdeccd952d4eed854f6003369a7cbb7b2836c694d${reset}"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$blue" "CHM DROPPER" "$reset" "$blue" "OBJINST" "$reset" "$blue" "POWERSHELL" "$reset"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$ink" "Hash-named single page" "$reset" "$ink" "Single-page dropper" "$reset" "$ink" "HTML Help auto-fire" "$reset"
    printf '  %b%-28s%b %b%-28s%b %b%s%b\n' \
        "$muted" "32a962…de61.chm" "$reset" "$muted" "32a962…de61.chm" "$reset" "$muted" "…otsinky.html:543" "$reset"
}

design_4() {
    concept "04" "EVIDENCE LEDGER"
    say "  ${bold}${ink}scan${reset}  ${muted}/ evidence ledger${reset}"
    brand_stripe
    blank
    printf '  %b01%b  %-74s %bHOSTILE 100%%%b\n' "$faint" "$reset" "meshcode-2.11.214-py3-none-any.whl" "$hostile" "$reset"
    say "      ${muted}sha256 96d7d3d3145b6b85062dff6362ef94fdb438375ed96f32754c0f159ecedb99b0 · known bad${reset}"
    printf '  %b%-4s %-48s %s%b\n' "$muted" "SEV" "TRAIT" "EVIDENCE" "$reset"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "Persistent Python remote terminal agent" "daemon.py:9"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "AI client config installs remote terminal agent" "run_agent.py:114"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "Python remote PTY control channel" "terminal_mirror_runner.py:6"
    blank
    printf '  %b02%b  %-74s %bHOSTILE 100%%%b\n' "$faint" "$reset" "32a962439ec0fb5559e494fe1ea6be039815d3c4c1cceb95b16dc123e5abde61.zip" "$hostile" "$reset"
    say "      ${muted}sha256 d43315ad7bbc22f97baddcacdeccd952d4eed854f6003369a7cbb7b2836c694d${reset}"
    printf '  %b%-4s %-48s %s%b\n' "$muted" "SEV" "TRAIT" "EVIDENCE" "$reset"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "Hash-named single-page CHM dropper" "32a962…de61.chm"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "Single-page CHM OBJINST dropper" "32a962…de61.chm"
    printf '  %b%-4s%b %-48s %s\n' "$hostile" "H" "$reset" "HTML Help Shortcut auto-fires PowerShell" "…otsinky.html:543"
}

design_5() {
    concept "05" "LITMUS"
    say "  ${bold}${ink}scan${reset}  ${muted}/ 2 files${reset}"
    brand_stripe
    blank
    say "  ${bold}${ink}meshcode-2.11.214-py3-none-any.whl${reset}  ${muted}WHL · known bad${reset}"
    printf '  %b HOSTILE %b  %b████████████████████ 100%%%b  %bknown bad%b\n' \
        "$hostile_badge" "$reset" "$hostile" "$reset" "$hostile" "$reset"
    say "  ${muted}sha256 96d7d3d3145b6b85062dff6362ef94fdb438375ed96f32754c0f159ecedb99b0${reset}"
    say "  ${hostile}● H${reset}  ${ink}Persistent Python remote terminal agent${reset}  ${faint}daemon.py:9${reset}"
    say "  ${hostile}● H${reset}  ${ink}AI client config installs remote terminal agent${reset}  ${faint}run_agent.py:114${reset}"
    say "  ${hostile}● H${reset}  ${ink}Python remote PTY control channel${reset}  ${faint}terminal_mirror_runner.py:6${reset}"
    blank
    say "  ${bold}${ink}32a962439ec0fb5559e494fe1ea6be039815d3c4c1cceb95b16dc123e5abde61.zip${reset}  ${muted}ZIP${reset}"
    printf '  %b HOSTILE %b  %b████████████████████ 100%%%b\n' \
        "$hostile_badge" "$reset" "$hostile" "$reset"
    say "  ${muted}sha256 d43315ad7bbc22f97baddcacdeccd952d4eed854f6003369a7cbb7b2836c694d${reset}"
    say "  ${hostile}● H${reset}  ${ink}Hash-named single-page CHM dropper${reset}  ${faint}32a962…de61.chm${reset}"
    say "  ${hostile}● H${reset}  ${ink}Single-page CHM OBJINST dropper${reset}  ${faint}32a962…de61.chm${reset}"
    say "  ${hostile}● H${reset}  ${ink}HTML Help Shortcut auto-fires PowerShell${reset}  ${faint}…otsinky.html:543${reset}"
    blank
    say "  ${faint}2 files · ${hostile}2 hostile${faint} · 0 clean · 11.0s${reset}"
}

usage() {
    say "usage: $0 [1|2|3|4|5|all]"
}

case "${1:-all}" in
    1) design_1 ;;
    2) design_2 ;;
    3) design_3 ;;
    4) design_4 ;;
    5) design_5 ;;
    all)
        design_1
        design_2
        design_3
        design_4
        design_5
        ;;
    -h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac

blank
