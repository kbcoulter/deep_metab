
# Nextflow Installation Guide

This guide installs **Nextflow locally in your home directory** using the official launcher.  
Installing Nextflow this way avoids issues with outdated conda packages and ensures compatibility with the cluster environment.

Nextflow requires **Java 17 or newer**.

---

## 1. Load Java

Load the Java module required by Nextflow.

```bash
module load java/17.0.10
````

Verify Java is available:

```bash
java -version
```

Expected output should contain something like:

```
openjdk version "17.0.10"
```

## 2. Install Nextflow

Create a personal `bin` directory and download the official Nextflow launcher.

```bash
mkdir -p $HOME/bin
cd $HOME/bin

curl -fsSL https://get.nextflow.io | bash
chmod +x nextflow
```

This installs the **Nextflow launcher**, which will automatically download the required runtime components.


## 3. Add Nextflow to Your PATH

Add your personal `bin` directory to your `PATH` so Nextflow can be run from anywhere.

```bash
export PATH=$HOME/bin:$PATH
```

To make this permanent, add the line above to your `~/.bashrc`.


## 4. Verify the Installation

Confirm that the correct executable is being used:

```bash
which nextflow
```

Expected output:

```
/home/<username>/bin/nextflow
```

Check the installed version:

```bash
nextflow -version
```

## 5. Pin a Specific Nextflow Version (Recommended)

For reproducible workflows, you can specify a fixed Nextflow version.

Example:

```bash
export NXF_VER=25.04.8
nextflow -version
```

This ensures all runs use the same version of Nextflow.

You may add this line to your `.bashrc` if you want it to be applied automatically.


## 6. Test Nextflow

Run a simple test pipeline:

```bash
nextflow run hello
```

If successful, Nextflow will download and execute a small demo workflow

## 7. Common Issues
### Script Not Running with Updated Values
Try renaming the file to a new file. Somehow, it works.

```bash
mv first.nf second.nf
nextflow clean -f
rm -rf work .nextflow*
nextflow run second.nf
```

### Getting a Java Error Version
You might see an error like the following:
```bash
ERROR: Cannot find Java or it's a wrong version -- please make sure that Java 17 or later (up to 25) is installed
NOTE: Nextflow is trying to use the Java VM defined by the following environment variables:
 JAVA_CMD: /usr/bin/java
 JAVA_HOME: 

```
If so, pleae load the correct java version if installed as described in [Step 1](#1-load-java).
