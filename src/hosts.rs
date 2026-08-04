//! Hosts in discovered references: URL host parsing, and the publisher-controlled
//! allowlist that keeps boilerplate out of the fetch phase.
//!
//! Every binary and script carries URLs that are documentation, not payload: a
//! license link, an XML namespace, a DTD, a vendor support page, and — in any
//! signed Mach-O or PE — the CRL and OCSP endpoints of the signing chain. A
//! scan of `/bin/ls` surfaces four such references before it surfaces anything
//! about `/bin/ls`. Fetching them costs a round trip each, fails noisily (they
//! 404, refuse, or return HTML), and cannot yield a sample: the hosts serve only
//! what their own publisher put there.
//!
//! So references to those hosts are dropped before the fetch phase sees them.
//! This is a *discovery* filter only — [`crate::pkg::run_url`] fetches whatever
//! URL the operator names, allowlist or not, because that is an explicit request
//! rather than an inference from a byte pattern.
//!
//! # What earns a place on the list
//!
//! A domain qualifies when **its entire DNS tree is controlled by one publisher
//! that serves only its own content**. Standards bodies, certificate
//! authorities, OS and silicon vendors, language projects, and API providers all
//! qualify: an attacker cannot put a second stage on `w3.org` or `crl.apple.com`.
//!
//! A domain is disqualified when *anyone* can publish a retrievable artifact
//! under it, however curated the process. That rules out code hosts
//! (github.com, gitlab.com, sourceforge.net), package registries (npmjs.com,
//! pypi.org, crates.io, maven.org, nuget.org), object storage and PaaS
//! (amazonaws.com, core.windows.net, githubusercontent.com, vercel.app),
//! model and dataset hubs (huggingface.co, kaggle.com), container registries
//! (docker.io, quay.io, ghcr.io), paste and file-drop services, sample
//! repositories (virustotal.com), archives that mirror arbitrary content
//! (archive.org), and URL shorteners, whose whole purpose is to point elsewhere.
//! None of those appear below, and none should be added.
//!
//! Several otherwise-qualifying publishers host third-party uploads on one
//! subdomain — Mozilla's add-ons, Arch's AUR, the Go module proxy, Maven
//! Central. [`THIRD_PARTY`] carves those back out and is checked first, so the
//! parent domain stays usable without opening a hole.
//!
//! The cost of a wrong entry is asymmetric: a missing domain wastes one HTTP
//! request, an over-broad one blinds the scanner to a real second stage. When a
//! candidate is arguable, leave it out.

/// Hosts under an allowlisted domain that *do* serve third-party uploads.
///
/// Checked before [`PUBLISHER`], so listing a parent domain there stays safe.
/// Every entry must sit under some [`PUBLISHER`] domain — an entry that doesn't
/// is dead weight (and usually a typo), which `exceptions_apply_to_a_listed_domain`
/// enforces.
const THIRD_PARTY: &[&str] = &[
    "addons.mozilla.org",          // user-submitted browser extensions
    "appexchange.salesforce.com",  // third-party Salesforce applications
    "apps.nextcloud.com",          // third-party Nextcloud apps
    "appsource.microsoft.com",     // third-party Microsoft business applications
    "aur.archlinux.org",           // Arch User Repository: unvetted PKGBUILDs
    "bugs.documentfoundation.org", // user-attached files on bug reports
    "bugs.freedesktop.org",        // user-attached files on bug reports
    "bugs.kde.org",                // user-attached files on bug reports
    "bugs.wireshark.org",          // user-attached files on bug reports
    "bugzilla.kernel.org",         // user-attached files on bug reports
    "bugzilla.mozilla.org",        // user-attached files on bug reports
    "build.opensuse.org",          // Open Build Service: user-built packages
    "cran.r-project.org",          // user-published R packages
    "exchange.adobe.com",          // third-party Adobe plugins
    "extensions.blender.org",      // user-submitted Blender extensions
    "extensions.gnome.org",        // user-submitted Shell extensions
    "extensions.libreoffice.org",  // user-submitted LibreOffice extensions
    "files.slack.com",             // user-uploaded workspace files
    "forge.puppet.com",            // user-published Puppet modules
    "forgeapi.puppet.com",         // user-published Puppet module archives
    "galaxy.ansible.com",          // user-published Ansible collections
    "gitlab.alpinelinux.org",      // user-hosted repositories
    "gitlab.freedesktop.org",      // user-hosted repositories
    "gitlab.gnome.org",            // user-hosted repositories
    "hackage.haskell.org",         // user-published Haskell packages
    "hub.docker.com",              // user-published container images
    "invent.kde.org",              // user-hosted repositories
    "issues.apache.org",           // user-attached files on issues
    "marketplace.atlassian.com",   // third-party Atlassian apps
    "marketplace.eclipse.org",     // third-party Eclipse solutions
    "marketplace.qt.io",           // third-party Qt components
    "pear.php.net",                // user-published PHP packages
    "pecl.php.net",                // user-published PHP extensions
    "people.apache.org",           // committer-controlled personal directories
    "people.debian.org",           // developer-controlled personal directories
    "people.freebsd.org",          // committer-controlled personal directories
    "people.freedesktop.org",      // developer-controlled personal directories
    "people.gnome.org",            // developer-controlled personal directories
    "pkg.julialang.org",           // user-published Julia packages
    "play.vuejs.org",              // user-authored playgrounds
    "plugins.jenkins.io",          // third-party Jenkins plugins
    "plugins.jetbrains.com",       // third-party IDE plugins
    "plugins.jquery.com",          // user-published jQuery plugins
    "proxy.golang.org",            // user-published Go modules
    "registry.terraform.io",       // user-published providers and modules
    "repo.maven.apache.org",       // Maven Central: user-published artifacts
    "savannah.gnu.org",            // project and user-hosted code repositories
    "splunkbase.splunk.com",       // third-party Splunk apps
    "storage.cloud.google.com",    // Cloud Storage objects
    "store.kde.org",               // user-submitted themes and widgets
    "sum.golang.org",              // Go checksum database
    "supermarket.chef.io",         // user-published Chef cookbooks
    "updates.jenkins.io",          // third-party Jenkins plugins
    "wiki.creativecommons.org",    // user-authored pages and uploads
];

/// Domains whose references in `/bin` are publisher/project documentation or
/// service endpoints rather than user-hosted artifacts. These are discovery
/// exceptions only; they do not change [`publisher_controlled`] or explicit
/// `scan url` behavior.
const DISCOVERY_DOMAINS: &[&str] = &[
    "ajax.googleapis.com",
    "bibnum.bnf.fr",
    "blog.clamav.net",
    "blogs.sun.com",
    "caca.zoy.org",
    "cb.vu",
    "cuddletech.com",
    "database.clamav.net",
    "docs.clamav.net",
    "docs.gluster.org",
    "download.salixos.org",
    "drobilla.net",
    "elfutils.org",
    "etherx.jabber.org",
    "fukuchi.org",
    "get.geojs.io",
    "gnupg.net",
    "greenwoodsoftware.com",
    "haque.net",
    "hwraid.le-vert.net",
    "icanhazip.com",
    "invisible-island.net",
    "ivarch.com",
    "kornshell.com",
    "lanana.org",
    "lib.openmpt.org",
    "libdv.sourceforge.net",
    "libimobiledevice.org",
    "llogiq.github.io",
    "lolengine.net",
    "mega-nerd.com",
    "milianw.de",
    "mirbsd.org",
    "nano-editor.org",
    "netpreserve.org",
    "ole.tange.dk",
    "openzfs.github.io",
    "rust-lang-nursery.github.io",
    "rust-lang.github.io",
    "schemas.openxmlformats.org",
    "smarden.org",
    "sodipodi.sourceforge.net",
    "spinics.net",
    "taper.alienbase.nl",
    "tealdeer-rs.github.io",
    "unicode-org.github.io",
    "upx.github.io",
    "web.dev",
    "www-s.acm.illinois.edu",
    "www.anandtech.com",
    "www.brendangregg.com",
    "www.cmu.edu",
    "www.corsair.com",
    "www.crucial.com",
    "www.edn.com",
    "www.floating-point-gui.de",
    "www.gluster.org",
    "www.google.com",
    "www.kvcd.net",
    "www.linux-usb.org",
    "www.lorenzobettini.it",
    "www.mbayer.de",
    "www.mjmwired.net",
    "www.oxygen-icons.org",
    "www.pcworld.com",
    "www.perl.com",
    "www.qrcode.com",
    "www.sdaoden.eu",
    "www.seafoam-labs.org",
    "www.skeeve.com",
    "www.surina.net",
    "www.tested.com",
    "www.tmpgenc.com",
    "www.tomshardware.com",
    "www.vesa.org",
    "www.yubi.co",
    "x265.readthedocs.org",
    "zbar.sourceforge.net",
    "zsh.sourceforge.net",
];

/// Exact URL references observed while scanning the system binaries in /bin.
///
/// These are discovery-only exceptions, not host allowlist entries. In
/// particular, an unseen URL on a code host remains fetchable, so this list
/// cannot hide a newly embedded second stage merely because it shares a host
/// with documentation found in a stock binary.
const DISCOVERY_URLS: &[&str] = &[
    r#"http://%s"#,
    r#"http://%s%s"#,
    r#"http://%s:%d"#,
    r#"http://%s:%d%s"#,
    r#"http://%s:%d/%s"#,
    r#"http://%s:%d/event/%u/%u"#,
    r#"http://[%s]:%d/%s"#,
    r#"http://askubuntu.com/questions/86483/how-can-i-see-or-change-default-run-level"#,
    r#"http://avisynth.nl/index.php/Select#SelectEvery"#,
    r#"http://bugs.librdf.org/"#,
    r#"http://comments.gmane.org/gmane.linux.file-systems.zfs.user/2032"#,
    r#"http://criu.org/Simple_loop"#,
    r#"http://github.com/jonhoo/inferno"#,
    r#"http://github.com/mharsch/arcstat"#,
    r#"http://github.com/zsh-users"#,
    r#"http://myserver.com/mysigs.ndb"#,
    r#"http://myserver/path/inxi"#,
    r#"http://osdir.com/ml/bug-gnu-utils-gnu/2010-12/msg00060.html"#,
    r#"http://other.url/"#,
    r#"http://portau/portaudio.com/"#,
    r#"http://purl.org/dc/dcmitype/StillImage"#,
    r#"http://purl.org/dc/elements/1.1/"#,
    r#"http://schemas.xmlsoap.or"#,
    r#"http://sf.net/p/lirc-remotes"#,
    r#"http://this.url/"#,
    r#"http://upx.sf.net"#,
    r#"http://wH9"#,
    r#"http://whatismyip.akamai.com/"#,
    r#"http://www."#,
    r#"http://www.brynosaurus.com/cachedir/"#,
    r#"http://www.mail-archive.com/linux-media@vger.kernel.org/msg56550.html"#,
    r#"http://www.spwg.org/salisbury_march_19_2002.pdf"#,
    r#"http://xmlns.com/foaf/0.1/Document"#,
    r#"https://%s%s"#,
    r#"https://..."#,
    r#"https://H9"#,
    r#"https://I9"#,
    r#"https://api.github.com/repos/AppImage"#,
    r#"https://api.github.com/repos/Seafoam-Labs/Shelly-ALPM/releases"#,
    r#"https://api.github.com/repos/Seafoam-Labs/shelly-icon-stream/releases/latest"#,
    r#"https://aur.archlinux.org"#,
    r#"https://aur.archlinux.org/"#,
    r#"https://aur.archlinux.org/cgit/aur.git/plain"#,
    r#"https://aur.archlinux.org/rpc/"#,
    r#"https://bugs.astron.com/"#,
    r#"https://bugs.freedesktop.org/show_bug.cgi?id=103807"#,
    r#"https://bugs.freedesktop.org/show_bug.cgi?id=34164"#,
    r#"https://buymeacoffee.com/zoeyerinba3"#,
    r#"https://cgit.git.savannah.gnu.org/cgit/parted.git/tree/AUTHORS"#,
    r#"https://clamav-site.s3.amazonaws.com/"#,
    r#"https://codeberg.org/ivarch/pv/issues"#,
    r#"https://codeberg.org/smxi/inxi"#,
    r#"https://codeberg.org/smxi/inxi^or^https://smxi.org/"#,
    r#"https://codeberg.org/smxi/inxi/raw/master/"#,
    r#"https://codeberg.org/smxi/inxi/raw/one/"#,
    r#"https://codeberg.org/smxi/inxi/raw/two/"#,
    r#"https://codeberg.org/smxi/pinxi/raw/master/"#,
    r#"https://crates.io/crates/bytecount"#,
    r#"https://crates.io/crates/regex"#,
    r#"https://criu.org/Deprecation"#,
    r#"https://dev.azure.com.useHttpPath"#,
    r#"https://docs.rs/cortex-m/0.6.3/cortex_m/asm/fn.wfi.html"#,
    r#"https://docs.rs/getrandom#nodejs-es-module-supportinternal_codedescriptionunknown_codeUnknown"#,
    r#"https://docs.rs/grep-printer/*/grep_printer/struct.JSON.html\fP."#,
    r#"https://docs.rs/parking_lot/latest/parking_lot/"#,
    r#"https://docs.rs/regex/1.*/regex/#syntax\fP"#,
    r#"https://docs.rs/regex/1.*/regex/bytes/index.html#syntax\fP"#,
    r#"https://docs.rs/regex/latest/regex/#avoid-re-compiling-regexes-especially-in-a-loop"#,
    r#"https://docs.rs/rustls/latest/rustls/manual/_03_howto/index.html#unexpected-eofTLS"#,
    r#"https://docs.rs/take_mut"#,
    r#"https://docs.rs/thiserror"#,
    r#"https://docs.rs/x86_64/0.12.2/x86_64/instructions/fn.hlt.html"#,
    r#"https://doi.org/10.5281/zenodo.21476260}"#,
    r#"https://en.wikipedia.org/wiki/Names_of_large_numbers"#,
    r#"https://en.wikipedia.org/wiki/Yoda_conditions"#,
    r#"https://flathub.org/api/v2"#,
    r#"https://flathub.org/repo/flathub.flatpakrepo"#,
    r#"https://fluxer.gg/vOjrMXcE"#,
    r#"https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda"#,
    r#"https://git.savannah.gnu.org/cgit/grep.git/tree/AUTHORS"#,
    r#"https://git.savannah.gnu.org/cgit/parallel.git/tree/doc/citation-notice-faq.txt"#,
    r#"https://github.com/AcademySoftwareFoundation/openexr/issues"#,
    r#"https://github.com/BurntSushi/ripgrep"#,
    r#"https://github.com/BurntSushi/ripgrep/discussions\fP."#,
    r#"https://github.com/BurntSushi/ripgrep/issues/2956"#,
    r#"https://github.com/BurntSushi/ripgrep/pull/2957"#,
    r#"https://github.com/BurntSushi/ripgrep\fP"#,
    r#"https://github.com/CachyOS/CachyOS-PKGBUILDS/issues"#,
    r#"https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md"#,
    r#"https://github.com/KhronosGroup/VulkanSC-Loader/blob/main/docs/LoaderDriverInterface.md"#,
    r#"https://github.com/Kobzol/cargo-pgo/blob/main/README.md"#,
    r#"https://github.com/Mindwerks/wildmidi"#,
    r#"https://github.com/Mindwerks/wildmidi/issues"#,
    r#"https://github.com/OpenVisualCloud/SVT-HEVC"#,
    r#"https://github.com/Perl/perl5/issues"#,
    r#"https://github.com/Perl/perl5/issues\n"#,
    r#"https://github.com/Seafoam-Labs/Shelly-ALPM"#,
    r#"https://github.com/aristocratos/btop"#,
    r#"https://github.com/beatgammit/base64-js/blob/master/index.js"#,
    r#"https://github.com/brendangregg/FlameGraph"#,
    r#"https://github.com/brendangregg/FlameGraph:"#,
    r#"https://github.com/cachyos/linux-cachyos.git"#,
    r#"https://github.com/clap-rs/clap/issues"#,
    r#"https://github.com/clap-rs/clap/issues.."#,
    r#"https://github.com/clap-rs/clap/issues/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/builder/arg.rs/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/util/flat_map.rs"#,
    r#"https://github.com/clap-rs/clap/issues/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/parser/arg_matcher.rs"#,
    r#"https://github.com/clap-rs/clap/issues/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/parser/matches/arg_matches.rs/usr/src/debug/rust/rustc-1.82.0-src/library/core/src/ops/function.rs/usr/src/debug/rust/rustc-1.82.0-src/library/alloc/src/vec/mod.rsUnknownFormatNoSuchFontInCollectionParseNoFilesystemIoNoSuchGlyphPlatformError/usr/src/debug/rust/rustc-1.82.0-src/library/core/src/str/pattern.rspartition_uefi-firmware-parser!"#,
    r#"https://github.com/clap-rs/clap/issues/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/parser/matches/matched_arg.rsa"#,
    r#"https://github.com/clap-rs/clap/issues=-/build/.cargo/registry/src/index.crates.io-6f17d22bba15001f/clap_builder-4.5.18/src/builder/arg.rsfalse--mid"#,
    r#"https://github.com/clap-rs/clap/issuesCommand::get_arg_conflicts_with:"#,
    r#"https://github.com/clap-rs/clap/issuesEmptyInvalidDigitPosOverflowNegOverflowNotAPowerOfTwo"#,
    r#"https://github.com/clap-rs/clap/issuesTryFromIntErrorErrorParseIntErrora"#,
    r#"https://github.com/clap-rs/clap/issuesUnknown"#,
    r#"https://github.com/clap-rs/clap/issuesa"#,
    r#"https://github.com/clap-rs/clap/issuesassertion"#,
    r#"https://github.com/clap-rs/clap/issuesattempt"#,
    r#"https://github.com/clap-rs/clap/issuesdescription"#,
    r#"https://github.com/clap-rs/clap/issuesinternal"#,
    r#"https://github.com/clap-rs/clap/issuesmid"#,
    r#"https://github.com/clap-rs/clap/issues{n}"#,
    r#"https://github.com/containers/crun/issues"#,
    r#"https://github.com/eza-community/eza"#,
    r#"https://github.com/facebook/zstd.git"#,
    r#"https://github.com/fastfetch-cli/fastfetch/raw/master/doc/json_schema.json"#,
    r#"https://github.com/fastfetch-cli/fastfetch/wiki/Color-Format-Specification"#,
    r#"https://github.com/fastfetch-cli/fastfetch/wiki/Configuration"#,
    r#"https://github.com/fastfetch-cli/fastfetch/wiki/Format-String-Guide."#,
    r#"https://github.com/fastfetch-cli/fastfetch/wiki/Logo-options"#,
    r#"https://github.com/freenas/freenas"#,
    r#"https://github.com/fribidi/fribidi/issues/new"#,
    r#"https://github.com/gnulib-modules/bootstrap/issues"#,
    r#"https://github.com/imaya/zlib.js"#,
    r#"https://github.com/jonhoo/inferno/"#,
    r#"https://github.com/kanaka/noVNC/blob/master/include/input.js"#,
    r#"https://github.com/libexpat/libexpat/issues"#,
    r#"https://github.com/libimobiledevice/libimobiledevice/issues"#,
    r#"https://github.com/libimobiledevice/libplist/issues"#,
    r#"https://github.com/libimobiledevice/libusbmuxd/issues"#,
    r#"https://github.com/llvm/llvm-project/issues/"#,
    r#"https://github.com/lsof-org/lsof"#,
    r#"https://github.com/lsof-org/lsof/blob/master/00FAQ"#,
    r#"https://github.com/lsof-org/lsof/blob/master/Lsof.8"#,
    r#"https://github.com/matthiaskrgr/rust-str-bench"#,
    r#"https://github.com/onetrueawk/awk"#,
    r#"https://github.com/openSUSE/catatonit/"#,
    r#"https://github.com/openzfs/zfs/blob/master/include/sys/zil.h"#,
    r#"https://github.com/openzfs/zfs/blob/master/module/zfs/arc.c"#,
    r#"https://github.com/phildawes/racer"#,
    r#"https://github.com/pkgconf/pkgconf"#,
    r#"https://github.com/polkit-org/polkit"#,
    r#"https://github.com/polkit-org/polkit#bugs-and-development"#,
    r#"https://github.com/ppp-project/ppp"#,
    r#"https://github.com/python/cpython/issues/83137."#,
    r#"https://github.com/rust-lang/cargo/issues/12918"#,
    r#"https://github.com/rust-lang/rfcs/blob/master/text/0212-restore-int-fallback.md"#,
    r#"https://github.com/rust-lang/rust-clippy"#,
    r#"https://github.com/rust-lang/rust-clippy/"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/10547"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/11085"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/11846#issuecomment-1820747924"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/3793"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/4295"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/5122"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/5953"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/6072"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/6353"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/6446"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/7306"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/8148"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/8241"#,
    r#"https://github.com/rust-lang/rust-clippy/issues/8496"#,
    r#"https://github.com/rust-lang/rust-clippy/pull/3123#issuecomment-422321765"#,
    r#"https://github.com/rust-lang/rust-clippy/pull/9484#issuecomment-1278922613"#,
    r#"https://github.com/rust-lang/rust/issues"#,
    r#"https://github.com/rust-lang/rust/issues/131154"#,
    r#"https://github.com/rust-lang/rust/issues/26925"#,
    r#"https://github.com/rust-lang/rust/issues/37612"#,
    r#"https://github.com/rust-lang/rust/issues/39465"#,
    r#"https://github.com/rust-lang/rust/issues/44690"#,
    r#"https://github.com/rust-lang/rust/issues/69826"#,
    r#"https://github.com/rust-lang/rust/pull/116505"#,
    r#"https://github.com/rust-lang/rustfmt/issues/3568"#,
    r#"https://github.com/sharkdp/fd/issuesexecsSIGHUPSIGINTSIGQUITSIGILLSIGTRAPSIGABRTSIGBUSSIGFPESIGKILLSIGUSR1SIGSEGVSIGUSR2SIGPIPESIGALRMSIGTERMSIGSTKFLTSIGCHLDSIGCONTSIGSTOPSIGTSTPSIGTTINSIGTTOUSIGURGSIGXCPUSIGXFSZSIGVTALRMSIGPROFSIGIOSIGPWRSIGSYSCtrlcTermination"#,
    r#"https://github.com/tldr-pages/tldr"#,
    r#"https://github.com/tldr-pages/tldr/releases/latest/download/Failed"#,
    r#"https://github.com/xiph/flac/issues"#,
    r#"https://github.com/zsh-users/zsh/blob/master/Etc/completion-style-guide"#,
    r#"https://gitlab.com/api/v4/projects/No"#,
    r#"https://gitlab.freedesktop.org/xdg/xdg-utils/-/issues"#,
    r#"https://gitlab.gnome.org/GNOME/glib/issues/new."#,
    r#"https://gitlab.gnome.org/GNOME/libxml2"#,
    r#"https://gitlab.gnome.org/GNOME/libxslt"#,
    r#"https://godbolt.org/z/ocrcenevq"#,
    r#"https://htop.dev/issues"#,
    r#"https://innoextract.constexpr.org/issues"#,
    r#"https://lame.sourceforge.io"#,
    r#"https://myserver.com/inxi"#,
    r#"https://openpgpkey.%s/.well-known/openpgpkey/%s/hu/%s?l=%s"#,
    r#"https://pagure.io/volume_key"#,
    r#"https://pastebin.ca"#,
    r#"https://pastebin.ca/quiet-paste.php?api="#,
    r#"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/refs/tags/2.2-prerelease-2024-08-08/sarif-2.2/schema/sarif-2-2.schema.json"#,
    r#"https://rt.cpan.org"#,
    r#"https://savannah.gnu.org/bugs/?36942"#,
    r#"https://savannah.gnu.org/bugs/?func=additem&group=wget."#,
    r#"https://savannah.gnu.org/projects/gettext"#,
    r#"https://sf.net/p/lirc-remotes/wiki"#,
    r#"https://slac...nl/people/alien/sbrepos/"#,
    r#"https://sourceforge.net/p/lirc-remotes/wiki/Checklist/"#,
    r#"https://sourceware.org/bugzilla"#,
    r#"https://sourceware.org/bugzilla/"#,
    r#"https://stackoverflow.com/q/13833088/363028"#,
    r#"https://superuser.com/questions/313447/seagate-momentus-xt-corrupting-files-linux-and-mac"#,
    r#"https://techpatterns.com/forums/forum-33.html"#,
    r#"https://todo.sr.ht/~kaniini/pkgconf"#,
    r#"https://unix.stackexchange.com/a/604943/2972"#,
    r#"https://unix.stackexchange.com/questions/439356/"#,
    r#"https://www."#,
    r#"https://www.clamav.net/about.html#credits"#,
    r#"https://www.clamav.net/presigned?type=fp"#,
    r#"https://www.clamav.net/presigned?type=malware"#,
    r#"https://www.clamav.net/reports/fp"#,
    r#"https://www.clamav.net/reports/malware"#,
    r#"https://www.clamav.net/reports/submit"#,
    r#"https://www.pastebin.ca"#,
    r#"https://www.pastebin.ca/upload.php"#,
    r#"https://www.pcre.org\fP"#,
    r#"https://www.reddit.com/r/rust/comments/3hb0wm/underhanded_rust_contest/cu5yuhr"#,
    r#"https://www.smartmontools.org/"#,
    r#"https://www.smartmontools.org/ticket/628"#,
    r#"https://www.smartmontools.org/wiki/SamsungF4EGBadBlocks"#,
    r#"https://www.youtube.com/watch?v=6c7pZYP_iIE"#,
    r#"https://yoursite.com/remote-ip"#,
    r#"https://yubi.co/ysa201701"#,
];

/// Domains whose whole DNS tree is one publisher serving its own content.
///
/// Matched on label boundaries against a reference's host, so an entry covers
/// the domain and every subdomain — `apple.com` covers `crl.apple.com` — but
/// never a lookalike registration like `apple.com.evil.test`.
///
/// Grouped by category and sorted within each group. See the module docs for
/// the inclusion criterion before adding anything.
const PUBLISHER: &[&str] = &[
    // ---- Standards bodies, specifications, and registries ----------------
    "3gpp.org",
    "ansi.org",
    "asyncapi.com",
    "c2pa.org",
    "cncf.io",
    "commonmark.org",
    "creativecommons.org",
    "cve.org",
    "cyclonedx.org",
    "dns.google",
    "dublincore.org",
    "ecma-international.org",
    "etsi.org",
    "first.org",
    "graphql.org",
    "gtld-servers.net",
    "iana.org",
    "icann.org",
    "iec.ch",
    "ieee.org",
    "ietf.org",
    "ipv4only.arpa",
    "iso.org",
    "itu.int",
    "json-ld.org",
    "json-schema.org",
    "jsonapi.org",
    "khronos.org",
    "mitre.org",
    "ntp.org",
    "oasis-open.org",
    "oauth.net",
    "omg.org",
    "openapis.org",
    "opencontainers.org",
    "opengroup.org",
    "openid.net",
    "opensource.org",
    "openssf.org",
    "owasp.org",
    "publicsuffix.org",
    "relaxng.org",
    "resolver.arpa",
    "rfc-editor.org",
    "root-servers.net",
    "schema.org",
    "semver.org",
    "sigstore.dev",
    "slsa.dev",
    "spdx.org",
    "swagger.io",
    "toml.io",
    "trustedcomputinggroup.org",
    "unicode.org",
    "w3.org",
    "whatwg.org",
    "xmlsoap.org",
    "yaml.org",
    // ---- Certificate authorities: the CRL/OCSP URLs in every signature ---
    "actalis.it",
    "amazontrust.com",
    "buypass.com",
    "camerfirma.com",
    "certificate-transparency.org",
    "certum.pl",
    "comodoca.com",
    "d-trust.net",
    "digicert.com",
    "entrust.com",
    "entrust.net",
    "geotrust.com",
    "globalsign.com",
    "globalsign.net",
    "godaddy.com",
    "identrust.com",
    "isrg.org",
    "lencr.org",
    "letsencrypt.org",
    "pki.goog",
    "quovadisglobal.com",
    "rapidssl.com",
    "sectigo.com",
    "ssl.com",
    "starfieldtech.com",
    "swisssign.com",
    "symantec.com",
    "telesec.de",
    "thawte.com",
    "usertrust.com",
    "verisign.com",
    "verisign.net",
    // ---- Public institutions and government ------------------------------
    "cdc.gov",
    "epa.gov",
    "europa.eu",
    "fda.gov",
    "nih.gov",
    "state.gov",
    "un.org",
    "unesco.org",
    "usda.gov",
    "whitehouse.gov",
    "who.int",
    "worldbank.org",
    // ---- Operating systems and distributions ------------------------------
    "almalinux.org",
    "alpinelinux.org",
    "android.com",
    "apple.com",
    "archlinux.org",
    "armbian.com",
    "buildroot.org",
    "cachyos.org",
    "canonical.com",
    "centos.org",
    "debian.org",
    "dragonflybsd.org",
    "fedoraproject.org",
    "freebsd.org",
    "fsf.org",
    "gentoo.org",
    "gnu.org",
    "haiku-os.org",
    "illumos.org",
    "kernel.org",
    "linux.com",
    "linuxfoundation.org",
    "linuxfromscratch.org",
    "lkml.org",
    "man7.org",
    "microsoft.com",
    "msftconnecttest.com",
    "msftncsi.com",
    "netbsd.org",
    "nixos.org",
    "openbsd.org",
    "openembedded.org",
    "openindiana.org",
    "opensuse.org",
    "openwrt.org",
    "opnsense.org",
    "pfsense.org",
    "qnx.com",
    "raspberrypi.com",
    "raspberrypi.org",
    "reactos.org",
    "redhat.com",
    "rockylinux.org",
    "slackware.com",
    "suse.com",
    "tldp.org",
    "ubuntu.com",
    "voidlinux.org",
    "windows.com",
    "windowsupdate.com",
    "yoctoproject.org",
    // ---- Silicon, hardware, and infrastructure vendors --------------------
    "altera.com",
    "amd.com",
    "analog.com",
    "arduino.cc",
    "arista.com",
    "arm.com",
    "bluetooth.com",
    "bluetooth.org",
    "broadcom.com",
    "cisco.com",
    "cypress.com",
    "dell.com",
    "espressif.com",
    "hp.com",
    "hpe.com",
    "ibm.com",
    "infineon.com",
    "intel.com",
    "juniper.net",
    "lenovo.com",
    "logitech.com",
    "marvell.com",
    "mediatek.com",
    "microchip.com",
    "netapp.com",
    "netgear.com",
    "nvidia.com",
    "nxp.com",
    "purestorage.com",
    "qnap.com",
    "qualcomm.com",
    "realtek.com",
    "renesas.com",
    "riscv.org",
    "seagate.com",
    "sifive.com",
    "silabs.com",
    "st.com",
    "supermicro.com",
    "synology.com",
    "ti.com",
    "tp-link.com",
    "ui.com",
    "wdc.com",
    "xilinx.com",
    // ---- Languages, runtimes, and toolchains ------------------------------
    "adoptium.net",
    "azul.com",
    "bazel.build",
    "bun.sh",
    "clojure.org",
    "cmake.org",
    "coffeescript.org",
    "common-lisp.net",
    "cppreference.com",
    "crystal-lang.org",
    "dart.dev",
    "deno.com",
    "elixir-lang.org",
    "elm-lang.org",
    "emscripten.org",
    "erlang.org",
    "flutter.dev",
    "fsharp.org",
    "gleam.run",
    "go.dev",
    "golang.org",
    "graalvm.org",
    "gradle.org",
    "groovy-lang.org",
    "haskell.org",
    "isocpp.org",
    "java.com",
    "jqlang.org",
    "julialang.org",
    "kotlinlang.org",
    "llvm.org",
    "lua.org",
    "luajit.org",
    "mesonbuild.com",
    "nim-lang.org",
    "ninja-build.org",
    "nodejs.org",
    "ocaml.org",
    "openjdk.org",
    "perl.org",
    "php.net",
    "purescript.org",
    "python.org",
    "r-project.org",
    "racket-lang.org",
    "ruby-lang.org",
    "rust-lang.org",
    "sbcl.org",
    "scala-lang.org",
    "scons.org",
    "swift.org",
    "tcl-lang.org",
    "tcl.tk",
    "translationproject.org",
    "tug.org",
    "typescriptlang.org",
    "vlang.io",
    "wasmtime.dev",
    "webassembly.org",
    "ziglang.org",
    // ---- Web frameworks and developer references -------------------------
    "ampproject.org",
    "angular.dev",
    "babeljs.io",
    "dot.net",
    "electronjs.org",
    "eslint.org",
    "getbootstrap.com",
    "getcomposer.org",
    "grpc.io",
    "jquery.com",
    "prettier.io",
    "protobuf.dev",
    "react.dev",
    "vite.dev",
    "vuejs.org",
    "webpack.js.org",
    // ---- Foundations and core open-source projects ------------------------
    "alsa-project.org",
    "apache.org",
    "blender.org",
    "boost.org",
    "brew.sh",
    "busybox.net",
    "bzip.org",
    "caddyserver.com",
    "chrony-project.org",
    "curl.se",
    "documentfoundation.org",
    "drobilla.net",
    "eclipse.org",
    "ffmpeg.org",
    "fsarchiver.org",
    "freedesktop.org",
    "gimp.org",
    "git-scm.com",
    "gnome.org",
    "gnupg.org",
    "gnutls.org",
    "graphicsmagick.org",
    "gtk.org",
    "haproxy.org",
    "haxx.se",
    "ijg.org",
    "imagemagick.org",
    "info-zip.org",
    "inkscape.org",
    "isc.org",
    "jenkins.io",
    "kde.org",
    "krita.org",
    "lib.openmpt.org",
    "libimobiledevice.org",
    "libjpeg-turbo.org",
    "libpng.org",
    "libreoffice.org",
    "libressl.org",
    "libsodium.org",
    "littlecms.com",
    "libvirt.org",
    "lighttpd.net",
    "live555.com",
    "linuxcontainers.org",
    "lz4.org",
    "mercurial-scm.org",
    "mozilla.org",
    "musl.libc.org",
    "neovim.io",
    "netfilter.org",
    "nginx.com",
    "nginx.org",
    "nlnetlabs.nl",
    "nmap.org",
    "notepad-plus-plus.org",
    "oberhumer.com",
    "opencv.org",
    "openexr.com",
    "openresty.org",
    "openssh.com",
    "openssl.org",
    "openvpn.net",
    "opus-codec.org",
    "pcre.org",
    "powerdns.com",
    "putty.org",
    "qemu.org",
    "quad9.net",
    "samba.org",
    "smxi.org",
    "speex.org",
    "squid-cache.org",
    "strongswan.org",
    "sublimetext.com",
    "systemd.io",
    "tcpdump.org",
    "tortoisesvn.net",
    "tukaani.org",
    "uclibc.org",
    "varnish-cache.org",
    "videolan.org",
    "virtualbox.org",
    "webmproject.org",
    "wireguard.com",
    "wireshark.org",
    "www.openoffice.org",
    "wxwidgets.org",
    "x.org",
    "xfce.org",
    "xiph.org",
    "zlib.net",
    // ---- Databases and data infrastructure --------------------------------
    "clickhouse.com",
    "cockroachlabs.com",
    "couchbase.com",
    "databricks.com",
    "duckdb.org",
    "elastic.co",
    "influxdata.com",
    "mariadb.com",
    "mariadb.org",
    "memcached.org",
    "mongodb.com",
    "mysql.com",
    "neo4j.com",
    "opensearch.org",
    "planetscale.com",
    "postgresql.org",
    "redis.com",
    "redis.io",
    "scylladb.com",
    "snowflake.com",
    "sqlite.org",
    "supabase.com",
    "teradata.com",
    "timescale.com",
    "valkey.io",
    "yugabyte.com",
    // ---- Cloud-native and DevOps projects ---------------------------------
    "ansible.com",
    "chef.io",
    "cilium.io",
    "consul.io",
    "containerd.io",
    "cri-o.io",
    "docker.com",
    "envoyproxy.io",
    "etcd.io",
    "fluentbit.io",
    "fluentd.org",
    "hashicorp.com",
    "helm.sh",
    "istio.io",
    "jaegertracing.io",
    "k8s.io",
    "kubernetes.io",
    "linkerd.io",
    "linux-kvm.org",
    "mosquitto.org",
    "nats.io",
    "openstack.org",
    "opentelemetry.io",
    "podman.io",
    "projectcalico.org",
    "prometheus.io",
    "proxmox.com",
    "puppet.com",
    "rabbitmq.com",
    "saltproject.io",
    "terraform.io",
    "traefik.io",
    "vagrantup.com",
    "xenproject.org",
    "zeromq.org",
    // ---- Enterprise and commercial software vendors -----------------------
    "adobe.com",
    "atlassian.com",
    "autodesk.com",
    "aws.amazon.com",
    "citrix.com",
    "cloud.google.com",
    "cloudflare.com",
    "datadoghq.com",
    "jetbrains.com",
    "jfrog.com",
    "newrelic.com",
    "nextcloud.com",
    "nutanix.com",
    "oracle.com",
    "pagerduty.com",
    "qt.io",
    "salesforce.com",
    "sap.com",
    "sentry.io",
    "servicenow.com",
    "slack.com",
    "splunk.com",
    "veeam.com",
    "vmware.com",
    "workday.com",
    "www.alibabacloud.com",
    // ---- Security vendors and research ------------------------------------
    "bitdefender.com",
    "checkpoint.com",
    "cisa.gov",
    "crowdstrike.com",
    "cyberark.com",
    "duosecurity.com",
    "eset.com",
    "fortinet.com",
    "kaspersky.com",
    "malwarebytes.com",
    "nist.gov",
    "okta.com",
    "onelogin.com",
    "paloaltonetworks.com",
    "pingidentity.com",
    "qualys.com",
    "rapid7.com",
    "sentinelone.com",
    "snyk.io",
    "sonarsource.com",
    "sophos.com",
    "tenable.com",
    "trendmicro.com",
    // ---- API providers: weather -------------------------------------------
    "accuweather.com",
    "aerisweather.com",
    "dwd.de",
    "met.no",
    "metoffice.gov.uk",
    "noaa.gov",
    "open-meteo.com",
    "openweathermap.org",
    "tomorrow.io",
    "visualcrossing.com",
    "weather.com",
    "weather.gov",
    "weatherapi.com",
    "weatherbit.io",
    "weatherstack.com",
    "wunderground.com",
    // ---- API providers: AI and machine learning ---------------------------
    "ai.google.dev",
    "anthropic.com",
    "api-docs.deepseek.com",
    "api.deepseek.com",
    "api.moonshot.ai",
    "api.z.ai",
    "assemblyai.com",
    "cohere.ai",
    "cohere.com",
    "dashscope-intl.aliyuncs.com",
    "dashscope-us.aliyuncs.com",
    "dashscope.aliyuncs.com",
    "deepgram.com",
    "deepmind.com",
    "docs.z.ai",
    "elevenlabs.io",
    "fireworks.ai",
    "generativelanguage.googleapis.com",
    "groq.com",
    "langchain.com",
    "llamaindex.ai",
    "maas.aliyuncs.com",
    "milvus.io",
    "mistral.ai",
    "openai.azure.com",
    "openai.com",
    "openrouter.ai",
    "perplexity.ai",
    "pinecone.io",
    "platform.kimi.ai",
    "qdrant.tech",
    "qianfan.baidubce.com",
    "runwayml.com",
    "stability.ai",
    "together.ai",
    "trychroma.com",
    "weaviate.io",
    "x.ai",
    // ---- API providers: geo, comms, payments, and data --------------------
    "alphavantage.co",
    "api.spotify.com",
    "brevo.com",
    "developer.spotify.com",
    "esri.com",
    "exchangerate-api.com",
    "finnhub.io",
    "here.com",
    "ip-api.com",
    "ipapi.co",
    "ipify.org",
    "ipinfo.io",
    "mailchimp.com",
    "mailgun.com",
    "mapbox.com",
    "maxmind.com",
    "nasa.gov",
    "nasdaq.com",
    "paypal.com",
    "plaid.com",
    "polygon.io",
    "postmarkapp.com",
    "restcountries.com",
    "sendgrid.com",
    "squareup.com",
    "stripe.com",
    "tencentcloudapi.com",
    "timeanddate.com",
    "tomtom.com",
    "twilio.com",
    "worldtimeapi.org",
    // ---- Documentation and reference --------------------------------------
    "devdocs.io",
    "linux.die.net",
    // ---- Reserved names that never resolve (RFC 2606, RFC 6761) -----------
    "example.com",
    "example.edu",
    "example.net",
    "example.org",
    "invalid",
    "localhost",
];

/// The host of a URL — scheme, path, query, and any `userinfo@` stripped. The
/// port is kept, so this reads as a human would write the authority.
///
/// Used both to label a redirect in the fetch log and to test a reference
/// against the allowlist. Only the authority survives, so a signed CDN URL's SAS
/// token or JWT never reaches a terminal or a log line.
pub(crate) fn host_of(url: &str) -> &str {
    let after_scheme = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host)
}

/// Whether `url` points at a host whose publisher is the only party that can
/// put content there — see the module docs for the criterion.
///
/// A `true` means the URL is boilerplate (a license link, a CRL endpoint, a
/// vendor doc page) and fetching it cannot produce a sample.
pub(crate) fn publisher_controlled(url: &str) -> bool {
    let host = normalize(host_of(url));
    if THIRD_PARTY.iter().any(|domain| under(&host, domain)) {
        return false;
    }
    PUBLISHER.iter().any(|domain| under(&host, domain))
}

/// Whether an exact URL was observed in /bin and should be suppressed only
/// when it is discovered in another file.
pub(crate) fn discovery_exception(url: &str) -> bool {
    let host = normalize(host_of(url));
    DISCOVERY_DOMAINS.iter().any(|domain| under(&host, domain)) || DISCOVERY_URLS.contains(&url)
}

/// Whether `host` *is* `domain` or sits beneath it, compared on label
/// boundaries — so `apple.com` covers `crl.apple.com` but not `evilapple.com`,
/// and never a lookalike like `apple.com.attacker.test`.
fn under(host: &str, domain: &str) -> bool {
    host.strip_suffix(domain)
        .is_some_and(|rest| rest.is_empty() || rest.ends_with('.'))
}

/// Lowercase a host and drop the two spellings that don't change which server it
/// names: a `:port` suffix and the trailing root dot. `WWW.Apple.COM.:443` and
/// `www.apple.com` are the same host, and both must match `apple.com`.
///
/// A bracketed IPv6 literal keeps its brackets and colons — no allowlist entry
/// is an IP, so it simply never matches.
fn normalize(host: &str) -> String {
    let host = match host.rsplit_once(':') {
        Some((h, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => h,
        _ => host,
    };
    host.strip_suffix('.').unwrap_or(host).to_ascii_lowercase()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn host_of_strips_scheme_path_query_and_userinfo() {
        assert_eq!(host_of("https://example.com/a/b?c=d#e"), "example.com");
        assert_eq!(
            host_of("https://release-assets.githubusercontent.com/x/y?sig=abc&jwt=xyz"),
            "release-assets.githubusercontent.com"
        );
        assert_eq!(host_of("https://example.com"), "example.com");
        assert_eq!(
            host_of("http://user:pass@host.test:8080/x"),
            "host.test:8080"
        );
        assert_eq!(host_of("bareword"), "bareword");
    }

    #[test]
    fn the_bin_ls_signature_urls_are_all_recognized() {
        // Verbatim from a scan of /bin/ls: the code-signing chain's CRL and OCSP
        // endpoints and the plist DTD. The trailing `0` on two of them is a DER
        // SEQUENCE byte the string extractor swept in with the URL — the host is
        // intact, so the allowlist still catches them.
        for url in [
            "http://www.apple.com/appleca/root.crl0",
            "http://crl.apple.com/codesigning.crl0",
            "http://www.apple.com/appleca/0",
            "http://www.apple.com/DTDs/PropertyList-1.0.dtd",
            "https://ocsp.apple.com/ocsp03-appleaica",
            "https://translationproject.org/team/",
        ] {
            assert!(publisher_controlled(url), "not filtered: {url}");
        }
    }

    #[test]
    fn discovery_exceptions_do_not_open_a_whole_host() {
        assert!(discovery_exception(
            "https://github.com/clap-rs/clap/issues"
        ));
        assert!(discovery_exception(
            "https://codeberg.org/smxi/inxi^or^https://smxi.org/"
        ));
        assert!(discovery_exception("https://www.mirbsd.org/mksh.htm"));
        assert!(discovery_exception("https://keys.gnupg.net/"));
        assert!(discovery_exception("https://www.kornshell.com/"));
        assert!(discovery_exception("https://www.spinics.net/lists/foo"));
        assert!(!discovery_exception(
            "https://github.com/clap-rs/clap/issues/999999"
        ));
        assert!(!discovery_exception("https://github.com/attacker/repo"));
    }

    #[test]
    fn subdomains_match_and_lookalikes_do_not() {
        assert!(publisher_controlled("https://crl.apple.com/x.crl"));
        assert!(publisher_controlled("https://apple.com/x"));
        // A label-boundary match, so neither a prefixed registration nor a
        // suffixed one gets in.
        assert!(!publisher_controlled("https://evilapple.com/stage2"));
        assert!(!publisher_controlled(
            "https://apple.com.attacker.test/stage2"
        ));
        assert!(!publisher_controlled("https://notw3.org/x"));
    }

    #[test]
    fn normalizes_case_port_and_root_dot() {
        assert!(publisher_controlled("https://WWW.Apple.COM/x"));
        assert!(publisher_controlled("https://www.apple.com.:443/x"));
        assert!(publisher_controlled(
            "http://python.org:80/ftp/python/README"
        ));
    }

    #[test]
    fn third_party_subdomains_override_their_parent() {
        // The parent is allowlisted...
        assert!(publisher_controlled("https://archlinux.org/packages/"));
        assert!(publisher_controlled("https://go.dev/dl/"));
        assert!(publisher_controlled("https://www.mozilla.org/MPL/2.0/"));
        // ...but the subdomain anyone can publish under is not.
        assert!(!publisher_controlled(
            "https://aur.archlinux.org/cgit/aur.git/snapshot/pkg.tar.gz"
        ));
        assert!(!publisher_controlled(
            "https://proxy.golang.org/example.com/m/@v/v1.0.0.zip"
        ));
        assert!(!publisher_controlled(
            "https://addons.mozilla.org/firefox/downloads/file/1/x.xpi"
        ));
    }

    #[test]
    fn global_publisher_and_api_domains_are_recognized() {
        for url in [
            "https://root-servers.net/",
            "https://dns.google/resolve?name=example.com",
            "https://www.who.int/health-topics/",
            "https://www.openoffice.org/download/",
            "https://api.spotify.com/v1/tracks/example",
            "https://developer.spotify.com/documentation/web-api/",
            "https://www.alibabacloud.com/help/en/model-studio/",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1/models",
            "https://api.deepseek.com/v1/models",
            "https://api-docs.deepseek.com/quick_start/pricing/",
            "https://api.z.ai/api/paas/v4/models",
            "https://docs.z.ai/api-reference/introduction",
            "https://api.moonshot.ai/v1/models",
            "https://platform.kimi.ai/docs/api/overview",
            "https://openrouter.ai/api/v1/models",
            "https://generativelanguage.googleapis.com/v1beta/models",
            "https://example.openai.azure.com/openai/models",
            "https://qianfan.baidubce.com/v2/models",
            "https://hunyuan.tencentcloudapi.com/",
        ] {
            assert!(publisher_controlled(url), "not filtered: {url}");
        }
    }

    #[test]
    fn mixed_publishers_and_hosted_content_are_not_allowlisted() {
        for url in [
            "https://people.freebsd.org/~user/stage2.bin",
            "https://galaxy.ansible.com/api/v3/plugin/ansible/content/published/collections/artifacts/x.tar.gz",
            "https://supermarket.chef.io/cookbook-versions/1/download",
            "https://forgeapi.puppet.com/v3/files/user-module-1.0.0.tar.gz",
            "https://registry.terraform.io/v1/providers/user/provider/versions",
            "https://extensions.blender.org/add-ons/example/",
            "https://extensions.libreoffice.org/assets/downloads/example.oxt",
            "https://bugzilla.kernel.org/attachment.cgi?id=1",
            "https://bugs.documentfoundation.org/attachment.cgi?id=1",
            "https://download.savannah.gnu.org/releases/project/stage2.tar.gz",
            "https://wiki.creativecommons.org/images/example.bin",
            "https://play.vuejs.org/#example",
            "https://www.vim.org/scripts/download_script.php?src_id=1",
            "https://en.wikipedia.org/w/index.php?title=X&action=raw",
            "https://api.openstreetmap.org/api/0.6/changeset/1",
            "https://stackoverflow.com/questions/1/example",
            "https://sourceware.org/pub/example/stage2.tar.gz",
        ] {
            assert!(!publisher_controlled(url), "wrongly filtered: {url}");
        }
    }

    #[test]
    fn hosts_that_serve_user_uploads_are_never_allowlisted() {
        // A regression fence: every one of these can serve an attacker-supplied
        // second stage, so none may ever be filtered out of the fetch phase.
        for url in [
            "https://github.com/o/r/releases/download/v1/x.tar.gz",
            "https://raw.githubusercontent.com/o/r/main/x.sh",
            "https://gist.github.com/o/abc/raw/x.sh",
            "https://gitlab.com/o/r/-/raw/main/x.sh",
            "https://bitbucket.org/o/r/raw/x.sh",
            "https://sourceforge.net/projects/p/files/x.zip",
            "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
            "https://files.pythonhosted.org/packages/x/y.whl",
            "https://pypi.org/simple/requests/",
            "https://rubygems.org/downloads/x.gem",
            "https://static.crates.io/crates/serde/1.0.0/download",
            "https://repo1.maven.org/maven2/x/y.jar",
            "https://www.nuget.org/api/v2/package/x",
            "https://my-bucket.s3.amazonaws.com/x.bin",
            "https://acct.blob.core.windows.net/c/x.bin",
            "https://storage.googleapis.com/b/x.bin",
            "https://cdn.discordapp.com/attachments/1/2/x.exe",
            "https://api.telegram.org/file/bot1:2/documents/x.exe",
            "https://huggingface.co/o/m/resolve/main/x.bin",
            "https://pastebin.com/raw/abcd",
            "https://cdn.jsdelivr.net/npm/x@1/y.js",
            "https://unpkg.com/x@1/y.js",
            "https://web.archive.org/web/2020/http://x.test/y.exe",
            "https://www.virustotal.com/api/v3/files/abc/download",
            "https://bit.ly/abc",
            "https://o.ngrok.io/x.bin",
            "https://o.herokuapp.com/x.bin",
            "https://o.vercel.app/x.bin",
            "https://o.github.io/x.bin",
            "https://f-droid.org/repo/x.apk",
            "https://community.chocolatey.org/api/v2/package/x",
        ] {
            assert!(!publisher_controlled(url), "wrongly filtered: {url}");
        }
    }

    #[test]
    fn no_duplicate_entries() {
        for (label, list) in [
            ("PUBLISHER", PUBLISHER),
            ("THIRD_PARTY", THIRD_PARTY),
            ("DISCOVERY_DOMAINS", DISCOVERY_DOMAINS),
        ] {
            let mut seen = HashSet::new();
            for domain in list {
                assert!(seen.insert(*domain), "{label} lists {domain} twice");
            }
        }
    }

    #[test]
    fn entries_are_bare_lowercase_domains() {
        for domain in PUBLISHER.iter().chain(THIRD_PARTY).chain(DISCOVERY_DOMAINS) {
            assert_eq!(
                *domain,
                domain.to_ascii_lowercase(),
                "{domain} must be lowercase to match a normalized host"
            );
            assert!(
                !domain.starts_with('.')
                    && !domain.ends_with('.')
                    && !domain.contains('/')
                    && !domain.contains(':'),
                "{domain} must be a bare domain, not a URL or a dotted prefix"
            );
        }
    }

    #[test]
    fn exceptions_apply_to_a_listed_domain() {
        // An exception under no allowlisted parent never fires, so it is either a
        // typo or a leftover from a removed entry. Either way it should not sit
        // here implying protection it isn't providing.
        for exception in THIRD_PARTY {
            assert!(
                PUBLISHER.iter().any(|domain| under(exception, domain)),
                "{exception} sits under no allowlisted domain"
            );
        }
    }

    #[test]
    fn entries_are_not_shadowed_by_a_broader_entry() {
        // `foo.example.com` alongside `example.com` is redundant: the parent
        // already covers it. Catches a list that has quietly grown duplicates in
        // two different category groups.
        for domain in PUBLISHER {
            let shadow = PUBLISHER
                .iter()
                .find(|other| *other != domain && under(domain, other));
            assert!(
                shadow.is_none(),
                "{domain} is already covered by {}",
                shadow.unwrap()
            );
        }
    }
}
