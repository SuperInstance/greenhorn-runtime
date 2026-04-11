package profiler

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

type Profile struct {
	OS       string
	Arch     string
	CPUCores int
	RAMMB    int64
	HasGPU   bool
	GPUName  string
	VRAMMB   int64
	Hostname string
}

func GetProfile() Profile {
	p := Profile{
		OS:       runtime.GOOS,
		Arch:     runtime.GOARCH,
		CPUCores: runtime.NumCPU(),
		Hostname: "unknown",
	}
	var si syscall.Sysinfo_t
	if syscall.Sysinfo(&si) == nil {
		p.RAMMB = int64(si.Totalram) / 1024 / 1024
	}
	if h, ok := syscall.Getenv("HOSTNAME"); ok {
		p.Hostname = h
	}
	p.HasGPU, p.GPUName, p.VRAMMB = detectGPU()
	return p
}

func detectGPU() (bool, string, int64) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output()
	if err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if len(lines) > 0 {
			parts := strings.Split(lines[0], ",")
			name := strings.TrimSpace(parts[0])
			vram := int64(0)
			if len(parts) > 1 {
				vram, _ = strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
			}
			return true, name, vram
		}
	}
	return false, "", 0
}
