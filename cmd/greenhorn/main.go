package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/SuperInstance/greenhorn-runtime/pkg/profiler"
	"github.com/SuperInstance/greenhorn-runtime/pkg/allocator"
	"github.com/SuperInstance/greenhorn-runtime/pkg/connector"
	"github.com/SuperInstance/greenhorn-runtime/pkg/scheduler"
	"github.com/SuperInstance/greenhorn-runtime/pkg/rigging"
)

func main() {
	token := flag.String("token", "", "GitHub PAT for fleet access")
	fleet := flag.String("fleet", "https://github.com/SuperInstance/greenhorn-onboarding", "Fleet onboarding repo")
	config := flag.String("config", "greenhorn.yaml", "Resource config file")
	riggingName := flag.String("rigging", "auto", "Rigging to deploy (auto/scout/coder/compute/thinker/scavenger)")
	flag.Parse()

	fmt.Println("🟢 Greenhorn Runtime v0.1.0")
	fmt.Println("   \"Download. Deploy. Specialize. Repeat.\"")
	fmt.Println()

	// 1. Profile the environment
	profile := profiler.GetProfile()
	fmt.Printf("Environment: %s/%s, %d cores, %dMB RAM", 
		profile.OS, profile.Arch, profile.CPUCores, profile.RAMMB)
	if profile.HasGPU {
		fmt.Printf(", GPU: %s (%dMB VRAM)", profile.GPUName, profile.VRAMMB)
	}
	fmt.Println()

	// 2. Load or auto-detect config
	resources, err := allocator.LoadConfig(*config)
	if err != nil {
		fmt.Printf("No config file (%v), auto-detecting...\n", err)
		resources = allocator.AutoFromProfile(&profile)
	}

	// 3. Apply token
	if *token != "" {
		os.Setenv("GITHUB_TOKEN", *token)
	}
	if os.Getenv("GITHUB_TOKEN") == "" {
		log.Fatal("No GITHUB_TOKEN. Use --token or set env var.")
	}

	// 4. Deploy rigging
	if *riggingName == "auto" {
		*riggingName = rigging.AutoSelect(&profile)
	}
	rig, err := rigging.Deploy(*riggingName, profile, resources)
	if err != nil {
		log.Fatalf("Failed to deploy rigging: %v", err)
	}
	fmt.Printf("Rigging: %s (%s)\n", rig.Name, rig.Description)

	// 5. Connect to fleet
	conn := connector.New(*fleet, os.Getenv("GITHUB_TOKEN"))
	fmt.Printf("Connecting to fleet: %s\n", *fleet)
	if err := conn.Connect(); err != nil {
		log.Fatalf("Fleet connection failed: %v", err)
	}
	fmt.Println("Connected. Reading fence board...")

	// 6. Start scheduler
	sched := scheduler.New(rig, resources, conn)
	sched.Start()

	// 7. Wait for shutdown signal
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	fmt.Println("\nParking rigging...")
	sched.Stop()
	rig.Park()
	fmt.Println("🟢 Greenhorn parked. See you next season.")
}
