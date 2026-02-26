// Command collect-benign downloads known-good samples from popular package sources.
//
// Sources: PyPI, NPM, Cargo, RubyGems, Go, Wolfi
//
// Usage:
//
//	collect-benign [--output ~/data/known-good] [--max 50] [--sources pypi,npm,cargo,wolfi]
package main

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const userAgent = "DIVINE-BenignCollector/1.0 (security research)"

var httpClient = &http.Client{
	Timeout: 5 * time.Minute, // Large files need more time
}

// Package lists - well-established, popular packages
var (
	pypiPackages = []string{
		"requests", "numpy", "pandas", "flask", "django", "pytest",
		"boto3", "pillow", "matplotlib", "scipy", "setuptools",
		"pip", "wheel", "cryptography", "pyyaml", "jinja2",
		"click", "sqlalchemy", "beautifulsoup4", "lxml", "aiohttp",
		"tornado", "celery", "redis", "psycopg2-binary", "pymongo",
		"httpx", "fastapi", "pydantic", "black", "mypy",
		"flake8", "pylint", "tox", "coverage", "sphinx",
		"docutils", "pygments", "markdown", "bleach", "html5lib",
		"chardet", "idna", "urllib3", "certifi", "six",
		"packaging", "attrs", "jsonschema", "pytz", "python-dateutil",
	}

	npmPackages = []string{
		"lodash", "express", "react", "axios", "moment",
		"chalk", "commander", "debug", "uuid", "async",
		"underscore", "yargs", "glob", "minimist", "semver",
		"inquirer", "dotenv", "cors", "body-parser", "cookie-parser",
		"morgan", "helmet", "passport", "jsonwebtoken", "bcrypt",
		"mongoose", "sequelize", "knex", "pg", "mysql2",
		"redis", "ioredis", "ws", "socket.io", "nodemailer",
		"cheerio", "puppeteer", "jest", "mocha", "chai",
		"eslint", "prettier", "typescript", "webpack", "babel-core",
		"rxjs", "ramda", "immutable", "date-fns", "dayjs",
	}

	cargoPackages = []string{
		"serde", "tokio", "clap", "rand", "regex",
		"log", "chrono", "anyhow", "thiserror", "lazy_static",
		"futures", "hyper", "reqwest", "actix-web", "rocket",
		"serde_json", "toml", "csv", "sha2", "aes",
		"rayon", "crossbeam", "parking_lot", "once_cell", "dashmap",
		"bytes", "walkdir", "glob", "ignore", "tracing",
		"env_logger", "ureq", "attohttpc", "uuid", "base64",
		"indexmap", "hashbrown", "smallvec", "bitflags", "num",
		"itertools", "either", "syn", "quote", "proc-macro2",
		"memchr", "aho-corasick", "bstr", "memmap2", "tempfile",
	}

	rubygemsPackages = []string{
		"rails", "rake", "bundler", "rspec", "sinatra",
		"puma", "sidekiq", "devise", "pundit", "nokogiri",
		"json", "faraday", "httparty", "rest-client", "pg",
		"mysql2", "sqlite3", "redis", "rspec-core", "minitest",
		"rubocop", "simplecov", "thor", "highline", "aws-sdk-core",
		"activesupport", "activerecord", "actionpack", "rack", "erubi",
		"i18n", "tzinfo", "concurrent-ruby", "connection_pool", "mail",
		"mime-types", "multipart-post", "addressable", "public_suffix", "domain_name",
		"http-cookie", "netrc", "unf", "ffi", "ethon",
		"typhoeus", "oj", "multi_json", "hashie", "virtus",
	}

	goPackages = []string{
		"github.com/gin-gonic/gin",
		"github.com/gorilla/mux",
		"github.com/labstack/echo/v4",
		"github.com/go-chi/chi/v5",
		"github.com/spf13/cobra",
		"github.com/spf13/viper",
		"github.com/urfave/cli/v2",
		"github.com/sirupsen/logrus",
		"go.uber.org/zap",
		"github.com/stretchr/testify",
		"github.com/go-playground/validator/v10",
		"gorm.io/gorm",
		"github.com/go-redis/redis/v8",
		"github.com/aws/aws-sdk-go",
		"google.golang.org/grpc",
		"github.com/prometheus/client_golang",
		"github.com/golang-jwt/jwt/v4",
		"golang.org/x/crypto",
		"golang.org/x/net",
		"golang.org/x/sys",
		"golang.org/x/text",
		"github.com/google/uuid",
		"github.com/pkg/errors",
		"github.com/mitchellh/mapstructure",
		"github.com/fatih/color",
		"github.com/olekukonko/tablewriter",
		"github.com/dgraph-io/badger/v3",
		"github.com/syndtr/goleveldb",
		"github.com/nats-io/nats.go",
		"github.com/segmentio/kafka-go",
	}

	// Wolfi packages - well-established system and language packages
	wolfiPackages = []string{
		"bash", "busybox", "ca-certificates", "curl", "git",
		"glibc", "openssl", "zlib", "libcrypto3", "libssl3",
		"python-3.12", "python-3.12-base", "py3-pip", "py3-setuptools",
		"nodejs", "npm", "go", "rust", "ruby",
		"gcc", "make", "binutils", "cmake", "pkgconf",
		"vim", "nano", "less", "file", "findutils",
		"coreutils", "grep", "sed", "gawk", "diffutils",
		"tar", "gzip", "bzip2", "xz", "zstd",
		"wget", "openssh", "gnupg", "apk-tools",
		"nginx", "redis", "postgresql", "sqlite",
		"jq", "yq", "ripgrep", "fd", "bat",
	}

	// Fedora packages - core system utilities (from Fedora 40)
	fedoraPackages = []string{
		"bash", "coreutils", "grep", "sed", "gawk",
		"tar", "gzip", "bzip2", "xz", "zstd",
		"curl", "wget", "openssh-clients", "gnupg2", "openssl",
		"python3", "perl", "ruby", "nodejs", "golang",
		"gcc", "make", "binutils", "cmake", "pkgconf",
		"vim-enhanced", "nano", "less", "file", "findutils",
		"git", "subversion", "mercurial",
		"nginx", "httpd", "redis", "postgresql", "sqlite",
		"jq", "ripgrep", "fd-find", "bat",
	}

	// Debian packages - core system utilities (from Debian stable)
	debianPackages = []string{
		"bash", "coreutils", "grep", "sed", "gawk",
		"tar", "gzip", "bzip2", "xz-utils", "zstd",
		"curl", "wget", "openssh-client", "gnupg", "openssl",
		"python3", "perl", "ruby", "nodejs", "golang",
		"gcc", "make", "binutils", "cmake", "pkg-config",
		"vim", "nano", "less", "file", "findutils",
		"git", "subversion", "mercurial",
		"nginx", "apache2", "redis", "postgresql", "sqlite3",
		"jq", "ripgrep", "fd-find", "bat",
	}

	// FreeBSD core sets - base OS distribution files
	freebsdSets = []string{
		"base.txz",   // Base system
		"kernel.txz", // Kernel
		"lib32.txz",  // 32-bit compatibility libraries
		"src.txz",    // Source code
		"tests.txz",  // Test suite
	}

	// FreeBSD ports - source tarballs from popular ports
	freebsdPorts = []string{
		"shells/bash",
		"editors/vim",
		"editors/nano",
		"devel/git",
		"www/nginx",
		"databases/redis",
		"databases/postgresql16-server",
		"lang/python3",
		"lang/ruby",
		"lang/go",
		"lang/rust",
		"security/openssl",
		"security/gnupg",
		"net/curl",
		"ftp/wget",
		"textproc/jq",
		"textproc/ripgrep",
		"sysutils/fd",
		"sysutils/bat",
		"archivers/gtar",
	}

	// OpenBSD base sets - core OS distribution files
	openbsdSets = []string{
		"base78.tgz",  // Base system
		"comp78.tgz",  // Compiler tools
		"man78.tgz",   // Manual pages
		"game78.tgz",  // Games
		"xbase78.tgz", // X11 base
		"xshare78.tgz", // X11 shared files
	}

	// NetBSD base sets - core OS distribution files
	netbsdSets = []string{
		"base.tar.xz",   // Base system
		"comp.tar.xz",   // Compiler tools
		"etc.tar.xz",    // Configuration files
		"games.tar.xz",  // Games
		"man.tar.xz",    // Manual pages
		"misc.tar.xz",   // Miscellaneous
		"rescue.tar.xz", // Rescue tools
		"text.tar.xz",   // Text processing
	}
)

func main() {
	homeDir, _ := os.UserHomeDir()
	defaultOutput := filepath.Join(homeDir, "data", "known-good")

	output := flag.String("output", defaultOutput, "Output directory")
	maxPkgs := flag.Int("max", 50, "Max packages per source")
	sourcesFlag := flag.String("sources", "all", "Comma-separated sources: pypi,npm,cargo,rubygems,go,all")
	workers := flag.Int("workers", 4, "Concurrent downloads per source")
	flag.Parse()

	sources := strings.Split(*sourcesFlag, ",")
	if contains(sources, "all") {
		sources = []string{"pypi", "npm", "cargo", "rubygems", "go", "wolfi", "fedora", "debian", "freebsd", "freebsd-ports", "openbsd", "netbsd"}
	}

	if err := os.MkdirAll(*output, 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	log.Printf("Output directory: %s", *output)

	total := 0
	for _, source := range sources {
		var count int
		var err error

		switch source {
		case "pypi":
			count, err = collectPyPI(*output, *maxPkgs, *workers)
		case "npm":
			count, err = collectNPM(*output, *maxPkgs, *workers)
		case "cargo":
			count, err = collectCargo(*output, *maxPkgs, *workers)
		case "rubygems":
			count, err = collectRubyGems(*output, *maxPkgs, *workers)
		case "go":
			count, err = collectGo(*output, *maxPkgs, *workers)
		case "wolfi":
			count, err = collectWolfi(*output, *maxPkgs, *workers)
		case "fedora":
			count, err = collectFedora(*output, *maxPkgs, *workers)
		case "debian":
			count, err = collectDebian(*output, *maxPkgs, *workers)
		case "freebsd":
			count, err = collectFreeBSD(*output, *maxPkgs, *workers)
		case "freebsd-ports":
			count, err = collectFreeBSDPorts(*output, *maxPkgs, *workers)
		case "openbsd":
			count, err = collectOpenBSD(*output, *maxPkgs, *workers)
		case "netbsd":
			count, err = collectNetBSD(*output, *maxPkgs, *workers)
		default:
			log.Printf("Unknown source: %s", source)
			continue
		}

		if err != nil {
			log.Printf("Error collecting from %s: %v", source, err)
		}
		log.Printf("Collected %d packages from %s", count, source)
		total += count
	}

	log.Printf("\nTotal: %d packages collected", total)
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func collectPyPI(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting PyPI packages...")
	destBase := filepath.Join(outputDir, "pypi")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := pypiPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(pkgName string) bool {
		// Get package info
		url := fmt.Sprintf("https://pypi.org/pypi/%s/json", pkgName)
		var data struct {
			Releases map[string][]struct {
				PackageType string `json:"packagetype"`
				URL         string `json:"url"`
			} `json:"releases"`
		}

		if err := fetchJSON(url, &data); err != nil {
			log.Printf("  PyPI/%s: %v", pkgName, err)
			return false
		}

		// Get stable versions
		var versions []string
		for v := range data.Releases {
			if !isPrerelease(v) {
				versions = append(versions, v)
			}
		}
		if len(versions) < 3 {
			return false
		}

		version := versions[len(versions)/2]
		releases := data.Releases[version]

		// Find sdist
		var sdistURL string
		for _, r := range releases {
			if r.PackageType == "sdist" {
				sdistURL = r.URL
				break
			}
		}
		if sdistURL == "" {
			return false
		}

		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", pkgName, version))
		if dirExists(pkgDir) {
			return true // Already have it
		}

		// Download and extract
		if err := downloadAndExtractTarGz(sdistURL, pkgDir); err != nil {
			log.Printf("  PyPI/%s: %v", pkgName, err)
			return false
		}

		log.Printf("✓ PyPI: %s-%s", pkgName, version)
		return true
	})
}

func collectNPM(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting NPM packages...")
	destBase := filepath.Join(outputDir, "npm")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := npmPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(pkgName string) bool {
		url := fmt.Sprintf("https://registry.npmjs.org/%s", pkgName)
		var data struct {
			Versions map[string]struct {
				Dist struct {
					Tarball string `json:"tarball"`
				} `json:"dist"`
			} `json:"versions"`
		}

		if err := fetchJSON(url, &data); err != nil {
			log.Printf("  NPM/%s: %v", pkgName, err)
			return false
		}

		var versions []string
		for v := range data.Versions {
			if !isPrerelease(v) {
				versions = append(versions, v)
			}
		}
		if len(versions) < 3 {
			return false
		}

		version := versions[len(versions)/2]
		tarball := data.Versions[version].Dist.Tarball
		if tarball == "" {
			return false
		}

		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", pkgName, version))
		if dirExists(pkgDir) {
			return true
		}

		if err := downloadAndExtractTarGz(tarball, pkgDir); err != nil {
			log.Printf("  NPM/%s: %v", pkgName, err)
			return false
		}

		log.Printf("✓ NPM: %s-%s", pkgName, version)
		return true
	})
}

func collectCargo(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting Cargo crates...")
	destBase := filepath.Join(outputDir, "cargo")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := cargoPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(crateName string) bool {
		url := fmt.Sprintf("https://crates.io/api/v1/crates/%s", crateName)
		var data struct {
			Versions []struct {
				Num    string `json:"num"`
				Yanked bool   `json:"yanked"`
				DlPath string `json:"dl_path"`
			} `json:"versions"`
		}

		if err := fetchJSON(url, &data); err != nil {
			log.Printf("  Cargo/%s: %v", crateName, err)
			return false
		}

		var stable []struct {
			Num    string
			DlPath string
		}
		for _, v := range data.Versions {
			if !v.Yanked && !isPrerelease(v.Num) {
				stable = append(stable, struct {
					Num    string
					DlPath string
				}{v.Num, v.DlPath})
			}
		}
		if len(stable) < 3 {
			return false
		}

		v := stable[len(stable)/2]
		dlURL := "https://crates.io" + v.DlPath

		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", crateName, v.Num))
		if dirExists(pkgDir) {
			return true
		}

		if err := downloadAndExtractTarGz(dlURL, pkgDir); err != nil {
			log.Printf("  Cargo/%s: %v", crateName, err)
			return false
		}

		log.Printf("✓ Cargo: %s-%s", crateName, v.Num)
		return true
	})
}

func collectRubyGems(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting RubyGems...")
	destBase := filepath.Join(outputDir, "rubygems")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := rubygemsPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(gemName string) bool {
		url := fmt.Sprintf("https://rubygems.org/api/v1/versions/%s.json", gemName)
		var versions []struct {
			Number     string `json:"number"`
			Prerelease bool   `json:"prerelease"`
		}

		if err := fetchJSON(url, &versions); err != nil {
			log.Printf("  RubyGems/%s: %v", gemName, err)
			return false
		}

		var stable []string
		for _, v := range versions {
			if !v.Prerelease {
				stable = append(stable, v.Number)
			}
		}
		if len(stable) < 3 {
			return false
		}

		version := stable[len(stable)/2]
		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", gemName, version))
		if dirExists(pkgDir) {
			return true
		}

		dlURL := fmt.Sprintf("https://rubygems.org/downloads/%s-%s.gem", gemName, version)

		// Download .gem file (it's a tar containing data.tar.gz)
		if err := downloadAndExtractGem(dlURL, pkgDir); err != nil {
			log.Printf("  RubyGems/%s: %v", gemName, err)
			return false
		}

		log.Printf("✓ RubyGems: %s-%s", gemName, version)
		return true
	})
}

func collectGo(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting Go modules...")
	destBase := filepath.Join(outputDir, "go")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := goPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(modulePath string) bool {
		escaped := strings.ReplaceAll(modulePath, "/", "%2F")
		listURL := fmt.Sprintf("https://proxy.golang.org/%s/@v/list", escaped)

		resp, err := httpGet(listURL)
		if err != nil {
			log.Printf("  Go/%s: %v", modulePath, err)
			return false
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		lines := strings.Split(strings.TrimSpace(string(body)), "\n")

		var versions []string
		for _, v := range lines {
			v = strings.TrimSpace(v)
			if v != "" && !isPrerelease(v) {
				versions = append(versions, v)
			}
		}
		if len(versions) < 2 {
			return false
		}

		version := versions[len(versions)/2]
		safeName := strings.ReplaceAll(modulePath, "/", "_")
		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", safeName, version))
		if dirExists(pkgDir) {
			return true
		}

		dlURL := fmt.Sprintf("https://proxy.golang.org/%s/@v/%s.zip", escaped, version)
		if err := downloadAndExtractZip(dlURL, pkgDir); err != nil {
			log.Printf("  Go/%s: %v", modulePath, err)
			return false
		}

		log.Printf("✓ Go: %s@%s", modulePath, version)
		return true
	})
}

func collectWolfi(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting Wolfi packages...")
	destBase := filepath.Join(outputDir, "wolfi")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	// First, fetch and parse the APKINDEX to get available packages and versions
	apkIndex, err := fetchWolfiIndex()
	if err != nil {
		log.Printf("  Failed to fetch Wolfi index: %v", err)
		return 0, err
	}

	packages := wolfiPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	return collectParallel(packages, workers, func(pkgName string) bool {
		// Find package in index
		pkg, ok := apkIndex[pkgName]
		if !ok {
			// Try to find packages that start with pkgName (e.g., "python-3.12" matches "python-3.12-base")
			for name, p := range apkIndex {
				if strings.HasPrefix(name, pkgName) {
					pkg = p
					pkgName = name
					ok = true
					break
				}
			}
		}
		if !ok {
			log.Printf("  Wolfi/%s: not found in index", pkgName)
			return false
		}

		pkgDir := filepath.Join(destBase, fmt.Sprintf("%s-%s", pkgName, pkg.Version))
		if dirExists(pkgDir) {
			return true
		}

		// Download from Wolfi repo
		dlURL := fmt.Sprintf("https://packages.wolfi.dev/os/x86_64/%s-%s.apk", pkgName, pkg.Version)

		if err := downloadAndExtractAPK(dlURL, pkgDir); err != nil {
			log.Printf("  Wolfi/%s: %v", pkgName, err)
			return false
		}

		log.Printf("✓ Wolfi: %s-%s", pkgName, pkg.Version)
		return true
	})
}

// WolfiPackage represents a package from the Wolfi APKINDEX
type WolfiPackage struct {
	Name    string
	Version string
}

// fetchWolfiIndex downloads and parses the Wolfi APKINDEX
func fetchWolfiIndex() (map[string]WolfiPackage, error) {
	url := "https://packages.wolfi.dev/os/x86_64/APKINDEX.tar.gz"
	resp, err := httpGet(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	gr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return nil, err
	}
	defer gr.Close()

	tr := tar.NewReader(gr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if header.Name == "APKINDEX" {
			return parseAPKIndex(tr)
		}
	}

	return nil, fmt.Errorf("APKINDEX not found in archive")
}

// parseAPKIndex parses the APKINDEX file format
func parseAPKIndex(r io.Reader) (map[string]WolfiPackage, error) {
	packages := make(map[string]WolfiPackage)
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}

	// APKINDEX format: key:value pairs separated by blank lines
	var currentPkg WolfiPackage
	for _, line := range strings.Split(string(data), "\n") {
		if line == "" {
			// End of package entry
			if currentPkg.Name != "" && currentPkg.Version != "" {
				packages[currentPkg.Name] = currentPkg
			}
			currentPkg = WolfiPackage{}
			continue
		}

		if len(line) < 2 || line[1] != ':' {
			continue
		}

		key := line[0]
		value := line[2:]

		switch key {
		case 'P': // Package name
			currentPkg.Name = value
		case 'V': // Version
			currentPkg.Version = value
		}
	}

	// Don't forget the last package
	if currentPkg.Name != "" && currentPkg.Version != "" {
		packages[currentPkg.Name] = currentPkg
	}

	return packages, nil
}

func collectFedora(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting Fedora packages...")
	destBase := filepath.Join(outputDir, "fedora")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := fedoraPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	// Fedora 42 x86_64 base URL
	baseURL := "https://dl.fedoraproject.org/pub/fedora/linux/releases/42/Everything/x86_64/os/Packages"

	return collectParallel(packages, workers, func(pkgName string) bool {
		// Fedora packages are organized by first letter
		firstLetter := strings.ToLower(pkgName[0:1])

		// Try to find the package - we need to guess the version or use repodata
		// For simplicity, we'll try common version patterns
		pkgDir := filepath.Join(destBase, pkgName)
		if dirExists(pkgDir) {
			return true
		}

		// Try to fetch the package listing page and find the package
		listURL := fmt.Sprintf("%s/%s/", baseURL, firstLetter)
		resp, err := httpGet(listURL)
		if err != nil {
			log.Printf("  Fedora/%s: %v", pkgName, err)
			return false
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		// Find a package that matches our name
		// Look for href="pkgName-version.fc40.x86_64.rpm" or similar
		lines := strings.Split(string(body), "\"")
		var rpmFile string
		for _, line := range lines {
			if strings.HasPrefix(line, pkgName+"-") && strings.HasSuffix(line, ".rpm") {
				// Avoid -devel, -doc, -libs variants unless that's what we asked for
				baseName := strings.TrimSuffix(line, ".rpm")
				parts := strings.Split(baseName, "-")
				if len(parts) >= 2 {
					// Check if this is the base package (not -devel, -libs, etc)
					nameEnd := parts[len(parts)-2] // version is last, release before that
					if !strings.HasSuffix(nameEnd, "devel") &&
						!strings.HasSuffix(nameEnd, "libs") &&
						!strings.HasSuffix(nameEnd, "doc") {
						rpmFile = line
						break
					}
				}
			}
		}

		if rpmFile == "" {
			log.Printf("  Fedora/%s: not found", pkgName)
			return false
		}

		dlURL := fmt.Sprintf("%s/%s/%s", baseURL, firstLetter, rpmFile)
		if err := downloadFile(dlURL, filepath.Join(pkgDir, rpmFile)); err != nil {
			log.Printf("  Fedora/%s: %v", pkgName, err)
			return false
		}

		log.Printf("✓ Fedora: %s", rpmFile)
		return true
	})
}

func collectDebian(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting Debian packages...")
	destBase := filepath.Join(outputDir, "debian")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	packages := debianPackages
	if len(packages) > maxPkgs {
		packages = packages[:maxPkgs]
	}

	// Use Debian's snapshot or main mirror
	// We'll use the API at sources.debian.org to find source packages
	return collectParallel(packages, workers, func(pkgName string) bool {
		pkgDir := filepath.Join(destBase, pkgName)
		if dirExists(pkgDir) {
			return true
		}

		// Try to get source package info from Debian API
		apiURL := fmt.Sprintf("https://sources.debian.org/api/src/%s/", pkgName)
		var data struct {
			Versions []struct {
				Version string `json:"version"`
				Suites  []string `json:"suites"`
			} `json:"versions"`
		}

		if err := fetchJSON(apiURL, &data); err != nil {
			log.Printf("  Debian/%s: %v", pkgName, err)
			return false
		}

		if len(data.Versions) == 0 {
			log.Printf("  Debian/%s: no versions found", pkgName)
			return false
		}

		// Find a stable version
		var version string
		for _, v := range data.Versions {
			for _, suite := range v.Suites {
				if suite == "bookworm" || suite == "stable" || suite == "bullseye" {
					version = v.Version
					break
				}
			}
			if version != "" {
				break
			}
		}
		if version == "" {
			version = data.Versions[0].Version // fallback to latest
		}

		// Download source tarball
		// Format: https://sources.debian.org/data/main/b/bash/5.2.15-2/bash_5.2.15.orig.tar.gz
		firstLetter := strings.ToLower(pkgName[0:1])
		if strings.HasPrefix(pkgName, "lib") && len(pkgName) > 4 {
			firstLetter = "lib" + strings.ToLower(pkgName[3:4])
		}

		// Try to get the file list for this version
		filesURL := fmt.Sprintf("https://sources.debian.org/api/src/%s/%s/", pkgName, version)
		var filesData struct {
			Package string `json:"package"`
			Version string `json:"version"`
		}
		if err := fetchJSON(filesURL, &filesData); err == nil {
			// Download the orig tarball if we can find it
			versionBase := strings.Split(version, "-")[0]
			tarballURL := fmt.Sprintf("https://deb.debian.org/debian/pool/main/%s/%s/%s_%s.orig.tar.gz",
				firstLetter, pkgName, pkgName, versionBase)

			if err := os.MkdirAll(pkgDir, 0755); err != nil {
				return false
			}

			if err := downloadAndExtractTarGz(tarballURL, pkgDir); err != nil {
				// Try .tar.xz
				tarballURL = strings.Replace(tarballURL, ".tar.gz", ".tar.xz", 1)
				if err := downloadAndExtractTarXz(tarballURL, pkgDir); err != nil {
					log.Printf("  Debian/%s: %v", pkgName, err)
					return false
				}
			}

			log.Printf("✓ Debian: %s-%s", pkgName, version)
			return true
		}

		return false
	})
}

func collectFreeBSD(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting FreeBSD base sets...")
	destBase := filepath.Join(outputDir, "freebsd")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	// FreeBSD 15.0 amd64 base URL
	baseURL := "https://download.freebsd.org/releases/amd64/15.0-RELEASE"

	sets := freebsdSets
	if len(sets) > maxPkgs {
		sets = sets[:maxPkgs]
	}

	return collectParallel(sets, workers, func(setName string) bool {
		destPath := filepath.Join(destBase, setName)

		// Skip if already downloaded
		if fileExists(destPath) {
			log.Printf("  FreeBSD/%s: already exists, skipping", setName)
			return true
		}

		dlURL := fmt.Sprintf("%s/%s", baseURL, setName)
		if err := downloadFile(dlURL, destPath); err != nil {
			log.Printf("  FreeBSD/%s: %v", setName, err)
			return false
		}

		log.Printf("✓ FreeBSD: %s", setName)
		return true
	})
}

func collectFreeBSDPorts(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting FreeBSD ports sources (via GNU mirrors)...")
	destBase := filepath.Join(outputDir, "freebsd-ports")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	// Use direct GNU/upstream URLs for common packages instead of parsing Makefiles
	gnuSources := map[string]string{
		"bash":      "https://ftp.gnu.org/gnu/bash/bash-5.2.tar.gz",
		"coreutils": "https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz",
		"grep":      "https://ftp.gnu.org/gnu/grep/grep-3.11.tar.xz",
		"sed":       "https://ftp.gnu.org/gnu/sed/sed-4.9.tar.xz",
		"gawk":      "https://ftp.gnu.org/gnu/gawk/gawk-5.3.0.tar.xz",
		"tar":       "https://ftp.gnu.org/gnu/tar/tar-1.35.tar.xz",
		"wget":      "https://ftp.gnu.org/gnu/wget/wget-1.24.5.tar.gz",
		"diffutils": "https://ftp.gnu.org/gnu/diffutils/diffutils-3.10.tar.xz",
		"findutils": "https://ftp.gnu.org/gnu/findutils/findutils-4.9.0.tar.xz",
		"gzip":      "https://ftp.gnu.org/gnu/gzip/gzip-1.13.tar.xz",
		"make":      "https://ftp.gnu.org/gnu/make/make-4.4.1.tar.gz",
		"patch":     "https://ftp.gnu.org/gnu/patch/patch-2.7.6.tar.xz",
		"bison":     "https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.xz",
		"flex":      "https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz",
		"nano":      "https://ftp.gnu.org/gnu/nano/nano-8.0.tar.xz",
		"vim":       "https://github.com/vim/vim/archive/refs/tags/v9.1.0000.tar.gz",
		"git":       "https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.44.0.tar.xz",
		"curl":      "https://curl.se/download/curl-8.6.0.tar.xz",
		"jq":        "https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-1.7.1.tar.gz",
		"openssl":   "https://www.openssl.org/source/openssl-3.2.1.tar.gz",
	}

	var sources []string
	for name := range gnuSources {
		sources = append(sources, name)
	}
	if len(sources) > maxPkgs {
		sources = sources[:maxPkgs]
	}

	return collectParallel(sources, workers, func(name string) bool {
		url := gnuSources[name]
		pkgDir := filepath.Join(destBase, name)
		if dirExists(pkgDir) {
			return true
		}

		if err := os.MkdirAll(pkgDir, 0755); err != nil {
			return false
		}

		var err error
		if strings.HasSuffix(url, ".tar.gz") {
			err = downloadAndExtractTarGz(url, pkgDir)
		} else if strings.HasSuffix(url, ".tar.xz") {
			err = downloadAndExtractTarXz(url, pkgDir)
		}
		if err != nil {
			log.Printf("  GNU-Sources/%s: %v", name, err)
			os.RemoveAll(pkgDir)
			return false
		}

		log.Printf("✓ GNU-Sources: %s", name)
		return true
	})
}

func collectOpenBSD(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting OpenBSD base sets...")
	destBase := filepath.Join(outputDir, "openbsd")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	// OpenBSD 7.8 amd64 base URL
	baseURL := "https://cdn.openbsd.org/pub/OpenBSD/7.8/amd64"

	sets := openbsdSets
	if len(sets) > maxPkgs {
		sets = sets[:maxPkgs]
	}

	return collectParallel(sets, workers, func(setName string) bool {
		destPath := filepath.Join(destBase, setName)

		// Skip if already downloaded
		if fileExists(destPath) {
			log.Printf("  OpenBSD/%s: already exists, skipping", setName)
			return true
		}

		dlURL := fmt.Sprintf("%s/%s", baseURL, setName)
		if err := downloadFile(dlURL, destPath); err != nil {
			log.Printf("  OpenBSD/%s: %v", setName, err)
			return false
		}

		log.Printf("✓ OpenBSD: %s", setName)
		return true
	})
}

func collectNetBSD(outputDir string, maxPkgs, workers int) (int, error) {
	log.Println("Collecting NetBSD base sets...")
	destBase := filepath.Join(outputDir, "netbsd")
	if err := os.MkdirAll(destBase, 0755); err != nil {
		return 0, err
	}

	// NetBSD 10.0 amd64 base URL
	baseURL := "https://cdn.netbsd.org/pub/NetBSD/NetBSD-10.0/amd64/binary/sets"

	sets := netbsdSets
	if len(sets) > maxPkgs {
		sets = sets[:maxPkgs]
	}

	return collectParallel(sets, workers, func(setName string) bool {
		destPath := filepath.Join(destBase, setName)

		// Skip if already downloaded
		if fileExists(destPath) {
			log.Printf("  NetBSD/%s: already exists, skipping", setName)
			return true
		}

		dlURL := fmt.Sprintf("%s/%s", baseURL, setName)
		if err := downloadFile(dlURL, destPath); err != nil {
			log.Printf("  NetBSD/%s: %v", setName, err)
			return false
		}

		log.Printf("✓ NetBSD: %s", setName)
		return true
	})
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// downloadFile downloads a file to the specified path, skipping if it already exists
func downloadFile(url, destPath string) error {
	// Skip if file already exists
	if fileExists(destPath) {
		return nil
	}
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return err
	}

	f, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

// downloadAndExtractTarXz downloads and extracts a .tar.xz file
func downloadAndExtractTarXz(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	// Download to temp file first, then use xz command to decompress
	tmp, err := os.CreateTemp("", "divine-*.tar.xz")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	tmp.Close()

	// Use tar with xz decompression
	return extractWithCommand(tmp.Name(), destDir, "xz")
}

// downloadAndExtractTarBz2 downloads and extracts a .tar.bz2 file
func downloadAndExtractTarBz2(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp("", "divine-*.tar.bz2")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	tmp.Close()

	return extractWithCommand(tmp.Name(), destDir, "bzip2")
}

// extractWithCommand extracts a compressed tar using an external command
func extractWithCommand(archivePath, destDir, compression string) error {
	var decompressFlag string
	switch compression {
	case "xz":
		decompressFlag = "-J"
	case "bzip2":
		decompressFlag = "-j"
	case "gzip":
		decompressFlag = "-z"
	default:
		return fmt.Errorf("unknown compression: %s", compression)
	}

	cmd := exec.Command("tar", decompressFlag, "-xf", archivePath, "-C", destDir)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// downloadAndExtractFreeBSDPkg downloads and extracts a FreeBSD .pkg file
func downloadAndExtractFreeBSDPkg(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	// FreeBSD .pkg files are tar+zstd or tar+xz archives
	// Try reading as tar+zstd first, fall back to tar+xz

	// Download to temp file
	tmp, err := os.CreateTemp("", "divine-*.pkg")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return err
	}
	tmp.Close()

	// Try to extract - FreeBSD pkgs are usually zstd compressed
	// For now, just save the raw file
	dest := filepath.Join(destDir, filepath.Base(url))
	return copyFile(tmp.Name(), dest)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	return err
}

// downloadAndExtractAPK downloads and extracts a Wolfi APK package
func downloadAndExtractAPK(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	// APK files are gzipped tarballs
	gr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gr.Close()

	tr := tar.NewReader(gr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Skip signature and control files, extract actual content
		name := header.Name
		if strings.HasPrefix(name, ".") {
			continue
		}

		target := filepath.Join(destDir, name)
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			continue
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			f, err := os.Create(target)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		case tar.TypeSymlink:
			// Handle symlinks
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			os.Remove(target) // Remove if exists
			if err := os.Symlink(header.Linkname, target); err != nil {
				// Ignore symlink errors (may point outside destDir)
				continue
			}
		}
	}
	return nil
}

func collectParallel(packages []string, workers int, fn func(string) bool) (int, error) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	var count int
	var mu sync.Mutex

	for _, pkg := range packages {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			if fn(p) {
				mu.Lock()
				count++
				mu.Unlock()
			}
		}(pkg)
	}

	wg.Wait()
	return count, nil
}

func fetchJSON(url string, v any) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return json.NewDecoder(resp.Body).Decode(v)
}

func httpGet(url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return resp, nil
}

func downloadAndExtractTarGz(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	gr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gr.Close()

	tr := tar.NewReader(gr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Sanitize path
		target := filepath.Join(destDir, header.Name)
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			continue // Skip files outside destDir
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			f, err := os.Create(target)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		}
	}
	return nil
}

func downloadAndExtractZip(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Download to temp file first (zip needs seeking)
	tmp, err := os.CreateTemp("", "divine-*.zip")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	defer tmp.Close()

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		return err
	}

	// Open as zip
	tmp.Seek(0, 0)
	stat, _ := tmp.Stat()
	zr, err := zip.NewReader(tmp, stat.Size())
	if err != nil {
		return err
	}

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	for _, f := range zr.File {
		target := filepath.Join(destDir, f.Name)
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			continue
		}

		if f.FileInfo().IsDir() {
			os.MkdirAll(target, 0755)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		out, err := os.Create(target)
		if err != nil {
			rc.Close()
			return err
		}

		io.Copy(out, rc)
		out.Close()
		rc.Close()
	}
	return nil
}

func downloadAndExtractGem(url, destDir string) error {
	resp, err := httpGet(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := os.MkdirAll(destDir, 0755); err != nil {
		return err
	}

	// .gem is a tar file containing data.tar.gz and metadata.gz
	tr := tar.NewReader(resp.Body)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if header.Name == "data.tar.gz" {
			gr, err := gzip.NewReader(tr)
			if err != nil {
				return err
			}

			dataTr := tar.NewReader(gr)
			for {
				h, err := dataTr.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					gr.Close()
					return err
				}

				target := filepath.Join(destDir, h.Name)
				if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
					continue
				}

				switch h.Typeflag {
				case tar.TypeDir:
					os.MkdirAll(target, 0755)
				case tar.TypeReg:
					os.MkdirAll(filepath.Dir(target), 0755)
					f, err := os.Create(target)
					if err != nil {
						gr.Close()
						return err
					}
					io.Copy(f, dataTr)
					f.Close()
				}
			}
			gr.Close()
			break
		}
	}
	return nil
}

func isPrerelease(version string) bool {
	v := strings.ToLower(version)
	return strings.Contains(v, "alpha") ||
		strings.Contains(v, "beta") ||
		strings.Contains(v, "rc") ||
		strings.Contains(v, "dev") ||
		strings.Contains(v, "canary") ||
		strings.Contains(v, "pre") ||
		strings.Contains(v, "experimental") ||
		strings.Contains(v, "nightly") ||
		strings.Contains(v, "snapshot") ||
		strings.Contains(v, "0.0.0")
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}
